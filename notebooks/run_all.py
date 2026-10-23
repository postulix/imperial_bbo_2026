import sys
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

notebooks = [
    'data_handling.ipynb',
    'f1.ipynb', 'f2.ipynb', 'f3.ipynb', 'f4.ipynb',
    'f5.ipynb', 'f6.ipynb', 'f7.ipynb', 'f8.ipynb',
]

for fname in notebooks:
    nb = nbformat.read(fname, as_version=4)
    client = NotebookClient(nb, timeout=300, kernel_name='python3',
                             resources={'metadata': {'path': '.'}},
                             allow_errors=False)
    status = 'SUCCESS'
    detail = ''
    try:
        client.execute()
    except CellExecutionError as e:
        status = 'STOPPED ON ERROR'
        detail = str(e).splitlines()[-1][:200]
    finally:
        nbformat.write(nb, fname)
    print(f'{fname}: {status} {detail}')
