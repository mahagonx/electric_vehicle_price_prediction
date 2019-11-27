# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 18:08:19 2019

@author: Matthias
"""

import scrapy

class QuotesSpider(scrapy.Spider):
    name = "quotes"

    def start_requests(self):
        urls = [
            'https://www.autoscout24.ch/de/autos/alle-marken?fuel=16&page=1&st=1&vehtyp=10',
            'https://www.autoscout24.ch/de/autos/alle-marken?fuel=16&page=2&st=1&vehtyp=10',
        ]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        page = response.url.split("/")[-2]
        filename = 'quotes-%s.html' % page
        with open(filename, 'wb') as f:
            f.write(response.body)
        self.log('Saved file %s' % filename)