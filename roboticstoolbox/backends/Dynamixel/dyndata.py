from bs4 import BeautifulSoup
import urllib.request
import re
from collections import namedtuple
import json

url = "https://emanual.robotis.com/docs/en/software/dynamixel/dynamixel_workbench/"

def table1_element(tables, name):
    try:
        value = tables[0].find(string=re.compile(name)).parent.find_next_siblings("td")[0]
        print(name, value.text)
        return value.text
    except Exception:
        print(name, ' not found')
        return None

def table2_element(tables, name, col=2):
    try:
        value = tables[1].find(string=re.compile(name)).parent.parent.find_next_siblings("td")[col]
        print(name, value.text)
        return value.text
    except Exception:
        print(name, ' not found')
        return None

def getint(s):
    print('getint: ', s)
    r = re.compile('[0-9,]+')
    m = r.search(s)
    if m is None:
        return None
    else:
        return int(m.group(0).replace(',', ''))

def getfloat(s):
    r = re.compile('[0-9,\.]+')
    m = r.search(s)
    return float(m.group(0).replace(',', ''))

with urllib.request.urlopen(url) as response:
    html = response.read()

    soup = BeautifulSoup(html, features="lxml")

    h = soup.find_all("h1", id="supported-dynamixel")
    table = h[0].find_next_sibling("table")
    models = []
    for tr in table.find_all('tr'):
        for a in tr.find_all('a'):
            print( a.text, "https://emanual.robotis.com" + a['href'])
            models.append((a.text, "https://emanual.robotis.com" + a['href']))

dyndata = namedtuple('dynamixel_data', 'model res posmin posmax')
dyndict = {}
for model, url in models:
    print('MODEL', model)
    with urllib.request.urlopen(url) as response:
        motor = {}
        motor['model'] = model
        motor['url'] = url

        html = response.read()

        soup = BeautifulSoup(html, features="lxml")
        tables = soup.find_all("table")

        # first table, find resolution
        res = table1_element(tables, "Resolution.*")
        r = res.split(' ')
        r[0] = getfloat(r[0])
        motor['resolution'] = r

        rd = table1_element(tables, "Running Degree.*")
        if rd is not None:
            # 0 [deg] ~ 300 [deg]
            rd = rd.split('~')
            motor['angular_range'] = [getint(x) for x in rd]

        # second table, find min/max position, model
        col = 2
        model = table2_element(tables, "Model.*")
        if model == '-':
            # some series (eg. X) have one less column in this table
            col = 1
            model = table2_element(tables, "Model.*", col)

        model = getint(model)
        motor['modelnum'] = model

        cw = table2_element(tables, "CW Angle Limit.*", col)
        if cw is not None:
            cw = getint(cw)
            ccw = table2_element(tables, "CCW Angle Limit.*", col)
            ccw = getint(ccw)
            motor['encoder_range'] = (cw, ccw)

        max = table2_element(tables[1], "Max.* Pos.*", col)
        if max is not None:
            max = int(max.text.replace(',', ''))

            min = table2_element(tables[1], "Min.* Pos.*", col)
            min = int(min.text.replace(',', ''))
            motor['encoder_range'] = (min, max)

        dyndict[model] = motor

print(dyndict)
with open('dynamixel.json', 'w') as outfile:
    json.dump(dyndict, outfile, indent=4)