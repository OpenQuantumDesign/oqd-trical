import os
import webbrowser

url = "file:///" + os.path.abspath("./build/html/index.html").replace("\\","/")
webbrowser.open_new_tab(url)