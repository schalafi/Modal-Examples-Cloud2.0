import os 
import pathlib
import modal 

FILE_DIR = os.path.dirname(__file__)
DIR = 'tmp/screenshots'

#Make sure screenshots dir exists
def create_screenshots_dir():
    SCREENSHOTS_DIR = os.path.join(FILE_DIR, DIR)
    print("SCREENSHOTS_DIR: ", SCREENSHOTS_DIR)
    if not os.path.exists(SCREENSHOTS_DIR):
        os.makedirs(SCREENSHOTS_DIR)


create_screenshots_dir()

stub = modal.Stub("example-screenshot")
#Run inside this file dir or outside :D
###RUN modal run screenshot.py --url https://www.youtube.com/watch?v=aeWyp2vXxqA
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
        os.path.join(FILE_DIR,
            "tmp/screenshots/screenshot.png"))
    print("Filename: ", filename)
    data = screenshot.call(url)
    filename.parent.mkdir(exist_ok=True)
    with open(filename, "wb") as f:
        f.write(data)
    print(f"wrote {len(data)} bytes to {filename}")

