from os import name as sys_name
from os.path import getsize
from downloader import download_file, get_headers, local_path

MATH_URL = 'https://math.stackexchange.com/questions/2582627/prove-sequent-using-natural-deduction'

def test_local_path():
    path = local_path(MATH_URL)
    assert path == 'prove-sequent-using-natural-deduction'
    if sys_name == 'posix':
        path = local_path(MATH_URL, '/tmp')
        assert path == '/tmp/prove-sequent-using-natural-deduction'
        path = local_path(MATH_URL, '/etc/passwd')
        assert path == '/etc/passwd'

def test_get_headers():
    fname = 'test_downloader.py'
    headers = get_headers(fname)
    size = getsize(fname)
    assert headers['Range'] == 'bytes=%d-' % size
