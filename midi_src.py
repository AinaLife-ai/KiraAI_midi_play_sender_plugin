"""MIDI 谱搜索与下载（httpx 异步版）。

渠道（2026-08-06 实测）：
  主搜索  : DuckDuckGo HTML 端点（免费无 key），site:bitmidi / site:hamienet 双路
  主下载  : BitMidi 歌曲页 → /uploads/数字.mid
  备选下载: HamieNET 直链 www.hamienet.com/{id}_{name}.mid（需 UA + Referer）

CLI 用法（便于独立调试）:
  python midi_src.py search <曲名>
  python midi_src.py download <页面URL> [输出路径]
"""
import asyncio
import contextvars
import functools
import html
import json
import re
import sys
import urllib.parse
from pathlib import Path

import httpx

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
BITMIDI_REFERER = "https://bitmidi.com/"
HAMIE_REFERER = "http://www.hamienet.com/"
DDG_ENDPOINT = "https://html.duckduckgo.com/html/"
UPLOAD_RE = re.compile(r"/uploads/(\d+\.mid)")

# 控制 httpx 是否读取环境变量代理（HTTP_PROXY/HTTPS_PROXY）。
# 云端 KiraAI 常配置代理且偶发不稳：网络错误时禁用代理直连重试一次，再切换渠道。
_use_proxy = contextvars.ContextVar("midi_src_use_proxy", default=True)


class MidiSrcError(Exception):
    pass


def _wrap_net_errors(ctx: str):
    """把 httpx 网络/状态异常统一转为可读 MidiSrcError（渠道链可跳过、LLM 可读）。

    策略：
    - 网络层错误（RequestError：Connect/Timeout/Proxy/Read 等）→ **禁用代理直连重试一次**
      （云端代理偶发不稳，直连可能成功），仍失败则转 MidiSrcError；
    - 状态码错误（HTTPStatusError）→ 直接转（重试无意义）；
    - 原始异常消息为空时（如 ConnectError）用异常类名兜底。
    """

    def deco(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            try:
                return await fn(*args, **kwargs)
            except MidiSrcError:
                raise
            except httpx.RequestError:
                # 网络错误：禁用代理（直连）重试一次
                token = _use_proxy.set(False)
                try:
                    return await fn(*args, **kwargs)
                except MidiSrcError:
                    raise
                except httpx.HTTPError as e2:
                    msg = str(e2).strip() or type(e2).__name__
                    raise MidiSrcError(
                        f"{ctx}网络错误：{msg}（已尝试代理与直连），已切换其他渠道") from e2
                finally:
                    _use_proxy.reset(token)
            except httpx.HTTPStatusError as e:
                msg = str(e).strip() or type(e).__name__
                raise MidiSrcError(f"{ctx}请求失败：{msg}，已尝试其他渠道") from e

        return wrapper

    return deco


def _client(timeout: float = 60.0) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        headers={"User-Agent": UA, "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8"},
        follow_redirects=True,
        timeout=timeout,
        trust_env=_use_proxy.get(),
    )


@_wrap_net_errors("DDG 搜索")
async def search_web(query: str, site: str = "", limit: int = 5) -> list[dict]:
    """DuckDuckGo HTML 搜索，返回 [{title, url}]。任何非 200 / 网络错误均抛 MidiSrcError（渠道链可跳过）。"""
    q = f"site:{site} {query}" if site else query
    async with _client() as c:
        resp = await c.get(DDG_ENDPOINT, params={"q": q})
        if resp.status_code != 200:
            raise MidiSrcError(f"DDG 搜索暂时不可用(HTTP {resp.status_code})，已尝试其他渠道")
        text = resp.text
    results = []
    # 结果形如：<a class="result__a" href="//duckduckgo.com/l/?uddg=<url>&rut=...">标题</a>
    for m in re.finditer(r'class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>', text, re.S):
        href, title = m.group(1), re.sub(r"<[^>]+>", "", m.group(2)).strip()
        url = href
        if "uddg=" in href:
            parsed = urllib.parse.urlparse(href)
            qs = urllib.parse.parse_qs(parsed.query)
            url = qs.get("uddg", [href])[0]
        url = html.unescape(url)
        if url.startswith("//"):
            url = "https:" + url
        if url.startswith("http") and url not in [r["url"] for r in results]:
            results.append({"title": title or query, "url": url})
            if len(results) >= limit:
                break
    return results


@_wrap_net_errors("BitMidi 下载")
async def bitmidi_download(page_url: str, out_path) -> str:
    """从 BitMidi 歌曲页提取 /uploads/数字.mid 并下载，返回保存路径。

    502 是 Cloudflare 临时错误，自动重试一次；其余非 200 抛 MidiSrcError。
    """
    async with _client() as c:
        c.headers["Referer"] = BITMIDI_REFERER
        page = await c.get(page_url)
        if page.status_code == 502:
            await asyncio.sleep(0.8)
            page = await c.get(page_url)
        if page.status_code != 200:
            raise MidiSrcError(f"谱页不可用(HTTP {page.status_code})：{page_url}")
        m = UPLOAD_RE.search(page.text)
        if not m:
            raise MidiSrcError(f"页面中未找到 MIDI 下载链接：{page_url}")
        dl_url = f"https://bitmidi.com/uploads/{m.group(1)}"
        async with c.stream("GET", dl_url) as resp:
            resp.raise_for_status()
            with open(out_path, "wb") as f:
                async for chunk in resp.aiter_bytes(8192):
                    f.write(chunk)
    return str(out_path)


@_wrap_net_errors("HamieNET 下载")
async def hamienet_download(page_url: str, out_path) -> str:
    """从 HamieNET 页面 URL 提取 {id}_{name}，拼直链下载（需 UA + Referer）。"""
    m = re.search(r"hamienet\.com/([0-9]+_[A-Za-z0-9_\-]+)\.(?:html?|mid)", page_url)
    if not m:
        raise MidiSrcError(f"无法从 URL 提取 HamieNET id_name：{page_url}")
    dl_url = f"http://www.hamienet.com/{m.group(1)}.mid"
    async with _client() as c:
        c.headers["Referer"] = HAMIE_REFERER
        resp = await c.get(dl_url)
        if resp.status_code != 200:
            raise MidiSrcError(f"HamieNET 下载失败：HTTP {resp.status_code}")
        with open(out_path, "wb") as f:
            f.write(resp.content)
    return str(out_path)


@_wrap_net_errors("BitMidi 搜索")
async def bitmidi_search(query: str, limit: int = 5) -> list[dict]:
    """BitMidi 站内搜索页。502 自动重试一次；其余非 200 / 网络错误抛 MidiSrcError（渠道链可跳过）。"""
    async with _client() as c:
        c.headers["Referer"] = BITMIDI_REFERER
        resp = await c.get("https://bitmidi.com/search", params={"q": query})
        if resp.status_code == 502:
            await asyncio.sleep(0.8)
            resp = await c.get("https://bitmidi.com/search", params={"q": query})
        if resp.status_code != 200:
            raise MidiSrcError(f"BitMidi 搜索暂时不可用(HTTP {resp.status_code})，已尝试其他渠道")
        text = resp.text
    results: list[dict] = []
    seen: set[str] = set()
    for m in re.finditer(r'href="/([a-z0-9-]+-mid)"', text):
        slug = m.group(1)
        if slug in seen:
            continue
        seen.add(slug)
        results.append({
            "title": slug[:-4].replace("-", " ").title(),
            "url": f"https://bitmidi.com/{slug}",
        })
        if len(results) >= limit:
            break
    return results


async def pick_midi(song: str, limit: int = 5) -> list[dict]:
    """按优先级收集候选谱（2026-08-06 实测）：

    1. BitMidi 站内搜索（最稳：URL 均为真实页面，不依赖第三方）
    2. DDG site:bitmidi / site:hamienet（补充）
    3. DDG 普通查询（偶发 202 挑战时忽略）
    4. bitmidi slug 直拼（最后兜底）

    返回 [{title, url}]。
    """
    cands: list[dict] = []
    try:
        cands += await bitmidi_search(song, limit)
    except MidiSrcError:
        pass
    for site in ("bitmidi.com", "hamienet.com"):
        try:
            cands += await search_web(song, site=site, limit=limit)
        except MidiSrcError:
            pass
    try:
        cands += await search_web(f"{song} midi", "", limit=limit)
    except MidiSrcError:
        pass
    if not cands:
        # slug 直拼兜底：bitmidi.com/{slug}-mid
        slug = re.sub(r"[^a-z0-9]+", "-", song.lower()).strip("-")
        if slug:
            cands.append({"title": song, "url": f"https://bitmidi.com/{slug}-mid"})
    seen: set[str] = set()
    out: list[dict] = []
    for c in cands:
        if c["url"] not in seen:
            seen.add(c["url"])
            out.append(c)
    return out[:limit]


async def try_download_any(cands: list[dict], out_path) -> tuple[str | None, str]:
    """逐个尝试下载候选，返回 (成功保存路径 或 None, 失败原因汇总)。"""
    errs: list[str] = []
    for c in cands:
        try:
            await download(c["url"], out_path)
            if is_valid_midi(out_path):
                return str(out_path), ""
            Path(out_path).unlink(missing_ok=True)
            errs.append(f"{c['url']} 文件无效")
        except Exception as e:
            errs.append(f"{c['url']} {e}")
    return None, "；".join(errs)


@_wrap_net_errors("直链下载")
async def direct_download(url: str, out_path) -> str:
    """任意直链/页面兜底下载：

    1) 直接 GET：响应即 MIDI（MThd 文件头）→ 直接保存（Google Drive 直链等适用）；
    2) 响应为 HTML 页面 → 提取页面内 .mid 链接（绝对/相对）再下载。
    502 临时错误自动重试一次；网络错误抛 MidiSrcError。
    """
    async with _client() as c:
        resp = await c.get(url)
        if resp.status_code == 502:
            await asyncio.sleep(0.8)
            resp = await c.get(url)
        if resp.status_code != 200:
            raise MidiSrcError(f"下载失败(HTTP {resp.status_code})：{url}")
        data = resp.content
        ctype = resp.headers.get("content-type", "")
        if data[:4] == b"MThd" or "html" not in ctype:
            # 直链命中：内容即 MIDI（或非 HTML 二进制）
            with open(out_path, "wb") as f:
                f.write(data)
            if data[:4] != b"MThd":
                raise MidiSrcError(f"下载内容不是有效 MIDI：{url}")
            return str(out_path)
        # HTML 页面 → 提取 .mid 链接
        text = data.decode("utf-8", errors="ignore")
        mid_url = ""
        m = re.search(r'https?://[^\s"\'<>]+\.mid\b', text)
        if m:
            mid_url = m.group(0)
        else:
            m2 = re.search(r'(?:href|src|data-url|action)="([^"]*\.mid)"', text, re.I)
            if m2:
                mid_url = m2.group(1)
        if not mid_url:
            raise MidiSrcError(f"页面中未找到 MIDI 下载链接：{url}")
        if mid_url.startswith("//"):
            mid_url = "https:" + mid_url
        elif mid_url.startswith("/"):
            base = urllib.parse.urlparse(url)
            mid_url = f"{base.scheme}://{base.netloc}{mid_url}"
        return await direct_download(mid_url, out_path)


async def download(page_url: str, out_path) -> str:
    """按 URL 自动选择源下载：
    bitmidi 歌曲页 → /uploads/ 提取；hamienet 页面 → id_name 直链；
    其余一律走 direct_download（任意直链 / Google Drive / 页面提取）。
    """
    if "hamienet.com" in page_url:
        return await hamienet_download(page_url, out_path)
    if "bitmidi.com" in page_url:
        return await bitmidi_download(page_url, out_path)
    return await direct_download(page_url, out_path)


def is_valid_midi(path) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(4) == b"MThd"
    except OSError:
        return False


async def _main(argv):
    if len(argv) < 2:
        print("用法: midi_src.py <search|download> <参数>")
        return
    cmd, arg = argv[0], argv[1]
    if cmd == "search":
        print(json.dumps(await pick_midi(arg), ensure_ascii=False, indent=1))
    elif cmd == "download":
        out = argv[2] if len(argv) > 2 else "midi_dl.mid"
        path = await download(arg, out)
        print(json.dumps({"path": path, "valid": is_valid_midi(path)}, ensure_ascii=False))
    else:
        print("未知命令: " + cmd)


if __name__ == "__main__":
    asyncio.run(_main(sys.argv[1:]))
