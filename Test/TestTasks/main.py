from Server.MainServer import main_server

main_server.run("127.0.0.1", 8380, debug=True, use_reloader=False)