fetch('https://api.ipify.org')
.then(response => response.text())
.then(ip => ip => IPython.notebook.kernel.execute('client_ip = "' + ip + '"'));
alert ('ip has being set');