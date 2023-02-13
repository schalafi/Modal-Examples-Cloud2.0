import os 
import pathlib
import modal 

def check_dir(dir_path:str):
    """
    dir_path: str
        check  of the dir exists 
        if not, it will create it.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

THIS_FILE_DIR = os.path.dirname(__file__)
DIR = 'tmp/screenshots'
DIR_PATH = os.path.join(THIS_FILE_DIR,DIR)

#Make sure screenshots dir exists
check_dir(DIR_PATH)

stub = modal.Stub("example-screenshot")
#Run inside this file dir or outside :D
###RUN: modal run screenshot.py --url https://www.youtube.com/watch?v=aeWyp2vXxqA
###Needs playwright locally

#Define the image (OS) with dependencies
image = modal.Image.debian_slim().run_commands(
    "apt-get install -y software-properties-common",
    "apt-add-repository non-free",
    "apt-add-repository contrib",
    "apt-get update",
    "pip install playwright==1.20.0",
    "playwright install-deps chromium",
    "playwright install chromium",
)

#Screenshot function
@stub.function(image=image)
async def screenshot(url):
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto(url, wait_until="networkidle")
        await page.screenshot(path="screenshot.png")
        await browser.close()
        data = open("screenshot.png", "rb").read()
        print("Screenshot of size %d bytes" % len(data))
        return data

#Use this entrypoint to run the screenshot function passing --url parameter.
@stub.local_entrypoint
def main(url: str = "https://modal.com"):

    filename = pathlib.Path(
        os.path.join(DIR_PATH,
            "screenshot.png"))
    print("Filename: ", filename)
    data = screenshot.call(url)
    filename.parent.mkdir(exist_ok=True)
    with open(filename, "wb") as f:
        f.write(data)
    print(f"wrote {len(data)} bytes to {filename}")

