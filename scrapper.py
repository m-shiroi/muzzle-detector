import gdown
from string import Template
from urllib import request

drive_link = Template("https://drive.google.com/uc?id=$id")
github_link = Template("https://raw.githubusercontent.com/$repo/$commit/$id")

def scrapper(itm, github_info):
    if itm["src"] == "github":
        name = itm["id"]
        link = github_link.substitute(**itm, **github_info)
        request.urlretrieve(link, name)
    elif itm["src"] == "drive":
        name = itm["out"]
        link = drive_link.substitute(**itm)
        gdown.download(link, name)
    else:
        raise Exception("ERRR")
