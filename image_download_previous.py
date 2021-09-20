import urllib.request
import urllib.parse
import re
import os

#create image path
def Imgpath(word):
    file_path = 'C:/Users/lijiaming3/Documents/grad_des_proj/image_' + word
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    else:
        file_path = file_path + '2'
        os.makedirs(file_path )
    print(file_path)
    return file_path

#get image url
def Imgurl(word):
    rep_list = []
    #Impersonate the browser to get the information and the target URL
    header = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.135 Safari/537.36',
        "referer": "https://image.baidu.com"
    }
    content= urllib.parse.quote(word,encoding='utf-8') #Identify Chinese
    for num in range(30,121,30):
        gsm = hex(num)[2:]         #Converts the decimal number num to hexadecimal number and takes the last two digits
        url = 'https://image.baidu.com/search/acjson?tn=resultjson_com&logid=9556938494902771516&ipn=rj&ct=201326592&is=&fp=result&' \
              'queryWord=' + content + '&cl=2&lm=-1&ie=utf-8&oe=utf-8&adpicid=&st=&z=&ic=&hd=&latest=&copyright=&' \
              'word=' + content + '&s=&se=&tab=&width=&height=&face=&istype=&qc=&nc=1&fr=&expermode=&force=&pn=' + str(num) + \
              '&rn=30&gsm=' + gsm +'&1619342915593='
        req = urllib.request.Request(url=url,headers=header)   #Obtain the request object
        page = urllib.request.urlopen(req).read()     #Request and read the return information
        try:     #pass if the info is not in uft-8
            response = page.decode('utf-8')       #parse data
            imgpattern = re.compile(r'"thumbURL":"(.*?)\.jpg')
            rsp_data = re.findall(imgpattern, response)
            rep_list += rsp_data
        except UnicodeDecodeError:
            pass
    return rep_list

# download images
def download_img(word):
    x = 1  # counting
    img_urllist = Imgurl(word)
    img_path = Imgpath(word)
    for url in img_urllist[:10]:
        pngurl = url.replace(r'"thumbURL":"', " ")
        print('pngurl', pngurl)
        path = img_path + '\\' + word + str(x) + '.png'  # path of image downloading
        pngdata = urllib.request.urlopen(pngurl).read()  # download data of image
        print('pngdata', pngdata)
        f = open(path, 'wb')  # write with binary
        f.write(pngdata)  # download image
        f.close()
        x = x + 1

if __name__ == '__main__':
    word = input("Enter the keyword of the pics that you want to scrape：")
    download_img(word)

