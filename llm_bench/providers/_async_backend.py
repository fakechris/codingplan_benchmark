"""
自定义 httpcore 异步网络后端
解决 Python 3.14 + anyio TLSStream.wrap 对部分服务器 (如 dashscope) 握手失败的问题
使用 asyncio 原生 SSL 连接代替 anyio 的 STARTTLS 方式
"""

from __future__ import annotations

import asyncio
import ssl
import typing

from httpcore._backends.base import AsyncNetworkBackend, AsyncNetworkStream, SOCKET_OPTION
from httpcore._exceptions import (
    ConnectError,
    ConnectTimeout,
    ReadError,
    ReadTimeout,
    WriteError,
    WriteTimeout,
)


class AsyncIOStream(AsyncNetworkStream):
    """基于 asyncio StreamReader/StreamWriter 的网络流"""

    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        self._reader = reader
        self._writer = writer

    async def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        try:
            return await asyncio.wait_for(
                self._reader.read(max_bytes),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            raise ReadTimeout() from None
        except OSError as e:
            raise ReadError(e) from e

    async def write(self, buffer: bytes, timeout: float | None = None) -> None:
        if not buffer:
            return
        try:
            self._writer.write(buffer)
            await asyncio.wait_for(self._writer.drain(), timeout=timeout)
        except asyncio.TimeoutError:
            raise WriteTimeout() from None
        except OSError as e:
            raise WriteError(e) from e

    async def aclose(self) -> None:
        try:
            self._writer.close()
            await self._writer.wait_closed()
        except OSError:
            pass

    async def start_tls(
        self,
        ssl_context: ssl.SSLContext,
        server_hostname: str | None = None,
        timeout: float | None = None,
    ) -> AsyncNetworkStream:
        # 直接使用 asyncio 的 start_tls
        try:
            transport = self._writer.transport
            protocol = transport.get_protocol()
            loop = asyncio.get_running_loop()
            new_transport = await asyncio.wait_for(
                loop.start_tls(transport, protocol, ssl_context, server_hostname=server_hostname),
                timeout=timeout,
            )
            self._writer._transport = new_transport  # type: ignore[attr-defined]
            return self
        except asyncio.TimeoutError:
            raise ConnectTimeout() from None
        except (OSError, ssl.SSLError) as e:
            raise ConnectError(e) from e

    def get_extra_info(self, info: str) -> typing.Any:
        if info == "ssl_object":
            return self._writer.get_extra_info("ssl_object")
        if info == "client_addr":
            return self._writer.get_extra_info("sockname")
        if info == "server_addr":
            return self._writer.get_extra_info("peername")
        if info == "socket":
            return self._writer.get_extra_info("socket")
        if info == "is_readable":
            return not self._reader.at_eof()
        return None


class AsyncIOBackend(AsyncNetworkBackend):
    """使用 asyncio 原生 API 的后端, 避免 anyio TLS 兼容性问题"""

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: typing.Iterable[SOCKET_OPTION] | None = None,
    ) -> AsyncNetworkStream:
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port, local_addr=(local_address, 0) if local_address else None),
                timeout=timeout,
            )
            if socket_options:
                raw_socket = writer.get_extra_info("socket")
                if raw_socket:
                    for option in socket_options:
                        raw_socket.setsockopt(*option)
            return AsyncIOStream(reader, writer)
        except asyncio.TimeoutError:
            raise ConnectTimeout() from None
        except OSError as e:
            raise ConnectError(e) from e

    async def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,
        socket_options: typing.Iterable[SOCKET_OPTION] | None = None,
    ) -> AsyncNetworkStream:
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_unix_connection(path),
                timeout=timeout,
            )
            return AsyncIOStream(reader, writer)
        except asyncio.TimeoutError:
            raise ConnectTimeout() from None
        except OSError as e:
            raise ConnectError(e) from e

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)
