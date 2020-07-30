from Server.HttpServer.MainServer import main_server

main_server.run("0.0.0.0", 8380, debug=True, use_reloader=False)