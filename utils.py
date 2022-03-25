from json import loads
from pathlib import Path
from pickle import UnpicklingError
import shelve
from shutil import which
from threading import Lock
from typing import cast

from requests import Session
from requests.adapters import HTTPAdapter


class Req:
    _data = cast(
        shelve.Shelf[bytes],
        shelve.open(Path(__file__).parent.joinpath("data.db").as_posix()),
    )
    _session = Session()
    _session.mount(
        "https://",
        # pool_connections: many different hosts, pool_maxsize: many same host
        HTTPAdapter(
            pool_connections=1000, pool_maxsize=1000, max_retries=3, pool_block=True
        ),
    )
    _data_lock = Lock()

    _curl_exe = which("curl") or "curl"

    @staticmethod
    def __call__(url, headers) -> bytes:
        key = str((url, headers))
        if key not in Req._data:
            res = Req._session.get(url, headers=headers).content
            with Req._data_lock:
                Req._data[key] = res
                Req._data.sync()
        try:
            with Req._data_lock:
                return Req._data[key]
        except UnpicklingError:
            print(f"UnpicklingError: {key=}")
            with Req._data_lock:
                del Req._data[key]
            return Req()(url, headers)


req = Req()


def req_json(url, headers) -> dict:
    return loads(req(url, headers))


def get_operon_data(operon_id: str):
    return req_json(
        f"https://wormbase.org/rest/widget/operon/{operon_id}/structure?download=1&content-type=application%2Fjson",
        None,
    )
