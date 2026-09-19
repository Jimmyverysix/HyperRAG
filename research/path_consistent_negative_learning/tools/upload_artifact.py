"""Upload a large research artifact over SFTP with size-based resume support.

The password is read from ``HYPERRAG_REMOTE_PASSWORD`` and is never written to
disk.  Source code is synchronized with Git; this utility is only for datasets
and model artifacts that do not belong in the repository.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path, PurePosixPath

import paramiko


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("local_path", type=Path)
    parser.add_argument("remote_path", type=PurePosixPath)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--user", required=True)
    parser.add_argument("--password-env", default="HYPERRAG_REMOTE_PASSWORD")
    return parser.parse_args()


def upload(args: argparse.Namespace) -> None:
    local_path = args.local_path.resolve(strict=True)
    local_size = local_path.stat().st_size
    password = os.environ.get(args.password_env)
    if not password:
        raise RuntimeError(f"环境变量 {args.password_env} 未设置")

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        args.host,
        port=args.port,
        username=args.user,
        password=password,
        timeout=30,
    )
    try:
        sftp = client.open_sftp()
        remote_name = str(args.remote_path)
        try:
            remote_size = sftp.stat(remote_name).st_size
        except FileNotFoundError:
            remote_size = 0

        if remote_size > local_size:
            raise RuntimeError(
                f"远端文件比本地文件大：remote={remote_size}, local={local_size}"
            )
        if remote_size == local_size:
            print(f"已完成，无需上传：{remote_name} ({local_size} bytes)", flush=True)
            return

        mode = "ab" if remote_size else "wb"
        transferred = remote_size
        next_report = transferred
        with local_path.open("rb") as source, sftp.open(remote_name, mode) as target:
            source.seek(remote_size)
            target.set_pipelined(True)
            while chunk := source.read(4 * 1024 * 1024):
                target.write(chunk)
                transferred += len(chunk)
                if transferred >= next_report:
                    percent = 100.0 * transferred / local_size
                    print(
                        f"{transferred}/{local_size} bytes ({percent:.1f}%)",
                        flush=True,
                    )
                    next_report = transferred + 32 * 1024 * 1024
        final_size = sftp.stat(remote_name).st_size
        if final_size != local_size:
            raise RuntimeError(
                f"上传后大小不一致：remote={final_size}, local={local_size}"
            )
        print(f"上传完成：{remote_name} ({final_size} bytes)", flush=True)
    finally:
        client.close()


def main() -> None:
    upload(parse_args())


if __name__ == "__main__":
    main()
