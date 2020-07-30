from Server.HttpServer.ClientServer import client_server

client_server.run("127.0.0.1", 6666, debug=True, use_reloader=False)