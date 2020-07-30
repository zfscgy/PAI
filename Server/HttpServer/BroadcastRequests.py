import threading
import requests


def broadcast_request(addrs: list, method: str, params=None, jsons=None):
    if method.lower() == "get":
        request = requests.get
    else:
        request = requests.post
    client_errs = dict()
    client_resps = [None] * len(addrs)
    request_threads = []

    def send_request(request_func, i, addr, **kwargs):
        try:
            resp = request_func(addr, **kwargs)
        except Exception as e:
            client_errs[i] = "Request failed: " + str(e)
            return
        if resp.status_code != requests.codes.ok:
            client_errs[i] = "Request get error code:" + str(resp.status_code)
        else:
            client_resps[i] = resp.json()
        return

    for i, addr in enumerate(addrs):
        if isinstance(params, list):
            param = params[i]
        else:
            param = params
        if isinstance(jsons, list):
            json_ = jsons[i]
        else:
            json_ = jsons
        kwargs = dict()
        if param is not None:
            kwargs['params'] = param
        if json_ is not None:
            kwargs['json'] = json_
        request_threads.append(threading.Thread(target=send_request, args=(request, i, addr), kwargs=kwargs))
        request_threads[-1].start()

    for request_thread in request_threads:
        request_thread.join()

    return client_errs, client_resps
