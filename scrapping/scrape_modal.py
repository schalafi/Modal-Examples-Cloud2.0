import re
import sys
import urllib.request
import modal


stub = modal.Stub(name="link-scraper")


@stub.function
def get_links(url):
    response = urllib.request.urlopen(url)
    html = response.read().decode("utf8")
    links = []
    for match in re.finditer('href="(.*?)"', html):
        links.append(match.group(1))
    return links


if __name__ == "__main__":
    #example url:https://www.redhat.com/en/topics/automation/what-is-a-webhook
    #replace this call with the modal remote call
    #links = get_links(sys.argv[1])
    with stub.run():
        links = get_links.call(sys.argv[1])
    print(links)
