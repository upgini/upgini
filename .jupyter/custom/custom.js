fetch('https://api.ipify.org')
.then(response => response.text())
.then(ip => IPython.notebook.kernel.execute('client_ip = "' + ip + '"'));