from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import uvicorn, aiohttp, asyncio
from io import BytesIO

from fastai import *
from fastai.vision import *

model_file_url = 'https://drive.google.com/file/d/14R6wEZczDZ3gh_9NH5BYWqBQiwyYe3f5/view?usp=sharing'
model_file_name = 'model'
classes = ['1016', '1018', '1020', '1023', '1027', '1029', '1030', '1037', '1138', '1197', '1200', '1208', '1223', '1240', '1242', '1243', '1244', '1245', '1251', '1264', '1267', '1278', '1286', '1296', '1299', '1300', '1310', '1312', '1315', '1324', '1329', '1333', '1334', '1343', '1355', '1356', '1372', '1374', '1379', '1381', '1384', '1385', '1388', '1389', '1391', '1392', '1410', '1415', '1422', '1424', '1426', '1431', '1435', '1446', '1462', '1467', '1469', '1470', '1482', '1487', '1602', '1613', '1618', '1619', '1624', '1625', '1626', '1627', '1631', '1632', '1636', '1647', '1661', '1688', '1691', '1692', '1693', '1705', '1706', '1707', '1732', '1755', '1756', '1791', '1792', '1830', '1831', '1832', '1833', '1834', '1835', '1883', '1888', '1889', '1890', '1891', '1892', '1893', '1894', '1914', '1915', '1922', '1938', '1939', '1940', '1941', '1942', '1943', '1944', '1945', '1959', '1960', '1961', '1968', '1969', '1972', '1980', '1990', '1991', '2088', '2181', '2182', '2183', '2190', '2194', '2195', '2299', '2332', '2333', '2334', '2335', '2336', '2380', '2411', '2453', '2524', '2525', '2530', '2542', '2554', '2555', '2556', '2560', '2563', '2572', '2599', '2602', '2603', '2606', '2619', '2620', '2625', '2643', '2665', '2670', '2671', '2699', '2705', '2729', '2746', '2756', '2761', '2774', '2803', '2865', '2888', '2920', '2921', '2923', '2925', '2926', '2927', '2929', '2930', '2944', '2948', '2986', '3020', '3031', '3062', '3065', '3111', '3112', '3141', '3147', '3181', '3183', '3194', '3195', '3218', '3219', '3242', '3273', '3306', '3310', '3312', '3318', '3324', '4061', '4081', '4088', '4124', '4133', '4142', '4143', '4145', '4180', '4217', '4218', '4238', '4280', '4281', '4327', '4328', '4375', '4376', '4460', '4483', '4484', '4533', '4534', '4536', '4537']
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))

async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)

async def setup_learner():
    await download_file(model_file_url, path/'models'/f'{model_file_name}.pth')
    data_bunch = ImageDataBunch.single_from_classes(path, classes, size=100).normalize(imagenet_stats)
    learn = cnn_learner(data_bunch, models.resnet34, pretrained=False)
    learn.load(model_file_name)
    return learn

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

@app.route('/')
def index(request):
    html = path/'view'/'index.html'
    return HTMLResponse(html.open().read())

@app.route('/analyze', methods=['POST'])
async def analyze(request):
    data = await request.form()
    img_bytes = await (data['file'].read())
    img = open_image(BytesIO(img_bytes))
    return JSONResponse({'result': learn.predict(img)[0]})

if __name__ == '__main__':
    if 'serve' in sys.argv: uvicorn.run(app, host='0.0.0.0', port=8080)

