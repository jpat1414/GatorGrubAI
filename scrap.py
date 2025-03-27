
from selenium import webdriver
path="C:\Users\User\Downloads\chromedriver_win32\chromedriver.exe"  #add chromedriver path
driver=webdriver.Chrome(path)
from time import sleep
from math import ceil
"""
~`@Image scrapper using selenium. Scrapping hashtag images from instagram
~`@for dataset creation.
`~@Just change your uername and password.
`!@Enter the hasthtag with an integer value limit.
@!`Rishabh Jain
"""
'\n~`@Image scrapper using selenium. Scrapping hashtag images from instagram\n~`@for dataset creation.\n`~@Just change your uername and password.\n`!@Enter the hasthtag with an integer value limit.\n@!`Rishabh Jain\n'
from selenium.webdriver.common.keys import Keys
import urllib
import os
driver.get('https://www.instagram.com/accounts/login')
username = driver.find_element_by_xpath('//*[@name="username"]')
password = driver.find_element_by_xpath('//*[@name="password"]')
username.send_keys("user")  #enter username
password.send_keys("pass") #enter password

sleep(3)

a=driver.find_element_by_css_selector("._5f5mN").click()
inp=raw_input("Enter the hashtag and limit: ")
inp=inp.split(" ")
inpu=inp[0]
limit=inp[-1]
Enter the hashtag and limit: tomcruise 200
#url="https://www.instagram.com/"+inpu+"/"
url="https://www.instagram.com/explore/tags/"+inpu+"/?hl=en"
driver.get(url)
#limit=200
scroll=0
b=[]
if int(limit)>20:
    scroll=int(ceil(int(limit)/20))
    

if scroll!=0:
    for i in range(scroll):
        htm=driver.find_element_by_tag_name("body")
        
        for j in driver.find_elements_by_css_selector(".FFVAD"):
            a=str(j.get_attribute("srcset").decode("utf-8"))
            a=a.split(",")
            b.append(a)
        htm.send_keys(Keys.END)
        print "scroll "+str(i+1)
        sleep(1)
sleep(3)
    

#print  len(driver.find_element_by_css_selector(".FFVAD"))


#print b

links=[]

for i in b:
    links.append(i[-1].split(" ")[0])
print links

links2=[]
for i in links:
    if len(i)<3:
        continue
    links2.append(i)
links2=list(set(links2))
scroll 1
scroll 2
scroll 3
scroll 4
scroll 5
scroll 6
scroll 7
scroll 8
scroll 9
scroll 10
['', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '']

count=0
os.system("mkdir C:\\Users\\User\\Desktop\\"+inpu)
os.chdir("C:\\Users\\User\\Desktop\\"+inpu)
for i in links2:
    f = open(str(count)+".jpg",'wb')
    f.write(urllib.urlopen(i).read())
    count+=1
    f.close()
 
