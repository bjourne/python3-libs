# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
from os.path import exists, getsize, isdir, join
from requests import get
from shutil import copyfileobj
from time import sleep
from urllib.parse import urlparse

def local_path(url, to = None):
    o = urlparse(url)
    doc_name = o.path.rsplit('/')[-1]
    if to is None:
        return doc_name
    elif isdir(to):
        return join(to, doc_name)
    else:
        return to

def get_headers(to):
    headers = {}
    if exists(to):
        headers['Range'] = 'bytes=%d-' % getsize(to)
    return headers

def format_bytes(content_length, old_size):
    if old_size:
        total = content_length + old_size
        frac = (old_size / total) * 100
        return '%d bytes, from %d%%' % (total, frac)
    return '%d bytes' % (content_length,)

def my_print(verbose, s):
    if verbose:
        print(s)

def write_to_file(r, to, chunk_size):
    if chunk_size > 0:
        with open(to, 'ab') as f:
            for chunk in r.iter_content(chunk_size = chunk_size):
                if chunk:
                    f.write(chunk)
    else:
        r.raw.decode_content = True
        with open(to, 'wb') as f:
            copyfileobj(r.raw, f)

def download_file(url, to = None,
                  retry_delay = 3, n_retries = 3, verbose = True,
                  assume_complete = False,
                  chunk_size = 256 * 1024):
    """Flexible and resilient file downloader.

    @assume_complete: if True, then the download is skipped without
    contacting the server if the file exists on disk and it's size is
    greater than zero.

    @chunk_size: Chunk size to use when streaming downloads. Set to 0
    to disable.
    """
    to = local_path(url, to)
    old_size = getsize(to) if exists(to) else 0
    if assume_complete and old_size > 0:
        my_print(verbose, '%s exists on disk - skipping' % to)
        return

    headers = get_headers(to)
    r = get(url, stream = True, headers = headers)
    content_length = r.headers.get('Content-Length')
    if not content_length or not content_length.isnumeric():
        if n_retries > 0:
            sleep(retry_delay)
            download_file(url, to, retry_delay, n_retries - 1)
            return
        raise Exception('Content-Length header missing!')
    content_length = int(content_length)

    target_size = old_size + content_length
    code = r.status_code
    if code == 416 or content_length == old_size:
        my_print(verbose, '%s downloaded - skipping' % to)
        return
    my_print(verbose, 'Downloading %s (%s) => %s' %
             (url, format_bytes(content_length, old_size), to))
    if code not in (200, 206):
        raise Exception('Some failure %s!' % r)
    write_to_file(r, to, chunk_size)
    curr_size = getsize(to)
    if curr_size < target_size:
        sleep(retry_delay)
        my_print(verbose, 'Only got %d bytes, retrying.' % curr_size)
        download_file(url, to)
