import os
import webbrowser

url = "file://" + os.path.realpath("./build/html/index.html")
webbrowser.open_new_tab(url)