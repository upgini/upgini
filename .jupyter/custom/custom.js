fetch('https://api.ipify.org')
.then(response => response.text())
.then(ip => IPython.notebook.kernel.execute('import os\nos.environ["CLIENT_IP"] = "' + ip + '"'));