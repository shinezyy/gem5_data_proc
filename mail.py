# -*- coding:utf8 -*-


import smtplib
import mimetypes
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header
from email.mime.image import MIMEImage


def send(sender, receiver, content):
    print 'Sending from', sender, 'to', receiver
    msg = MIMEMultipart()
    msg['From'] = Header(sender)
    msg['To'] = Header(receiver)
    msg['Subject'] = Header('gem5 stat', 'utf-8')
    msg.attach(MIMEText(content, 'plain', 'utf-8'))

    passwd = ''
    with open(os.path.expanduser('~/projects/password')) as f:
        for line in f:
            passwd = line.strip('\n')

    smtp = smtplib.SMTP()
    try:
        smtp.connect('smtp.163.com', '25')
        smtp.login(sender, passwd)
        smtp.sendmail(sender, receiver, msg.as_string())
        smtp.quit()
        print 'Sent'
    except smtplib.SMTPException, err:
        print 'Failed'
        print err
