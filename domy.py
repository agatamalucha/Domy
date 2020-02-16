# -*- coding: utf-8 -*-
import scrapy
from scrapy import Spider
from selenium import webdriver
from scrapy.selector import Selector
from scrapy.loader import ItemLoader
from apt.items import AptItem

class DomySpider(scrapy.Spider):
	name = 'domy'
	allowed_domains = ['otodom.pl']
	start_urls = ['http://otodom.pl/']


	def parse(self, response):
		self.driver=webdriver.Chrome('D:/Panda_projects/Wroclaw/chromedriver')
		self.driver.get('https://www.otodom.pl/sprzedaz/wroclaw/?search%5Bdescription%5D=1&search%5Bregion_id%5D=1&search%5Bsubregion_id%5D=381&search%5Bcity_id%5D=39')
		sel = Selector(text =self.driver.page_source)
		content = sel.xpath('//div[@class="col-md-content section-listing__row-content"]')
		offers=content.xpath('.//div[@class="offer-item-details"]')
	
		for offer in offers:
			l=ItemLoader(item=AptItem(),selector= offer)
			url=offer.xpath('.//h3/a/@href').extract_first() 
			location=offer.xpath('.//p/text()').extract_first()

			details=offer.xpath('.//ul/li/text()').extract()
			if len(details)==4:
				number_rooms=offer.xpath('.//ul/li/text()').extract()[0]
				area=offer.xpath('.//ul/li/text()').extract()[2]
				price_per_meter=offer.xpath('.//ul/li/text()').extract()[3]



			l.add_value('url',url)
			l.add_value('location',location)
			l.add_value('number_rooms',number_rooms)
			l.add_value('area',area)
			l.add_value('price_per_meter',price_per_meter)
		
			yield l.load_item()

# https://www.otodom.pl/sprzedaz

