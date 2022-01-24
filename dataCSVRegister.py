'''dataCSVRegister is a tool for register data from a dictionary type variable to CSV file.
All right reserved to veykor(https://github.com/veykor)
Version:1.0
'''

from vcommon import *

class dataCSVRegister:
	fileName = None
	headers=None
	data = {}
	
	def __init__(self, fileName='data.csv'):
		self.fileName=fileName
		
	def clearData(self):
		self.data.clear()
		
	def registerDict(self,dictIn):
		self.data.update(dictIn)
		
	def set(self, **paramsIn):
		for param in paramsIn.keys():
			self.data[param] = paramsIn[param]
			
	def get(self, *paramsOut):
		response=[]
		for param in paramsOut:
			data = self.data.get(param)
			response.append(data)
		return response
	
	def dataCSVFormatter(self,dataList):
		dataFormatted = ''
		for data in dataList:
			dataFormatted += f'{data};'
		dataFormatted = f'{dataFormatted[:-1]}\n'
		return dataFormatted			
	
	def createCSVHeaders(self):
		self.headers = self.dataCSVFormatter(self.data.keys())
	
	def createCSVData(self):
		return self.dataCSVFormatter(self.data.values())		
	
	def registerNewLine(self):
		with open(self.fileName,"a") as file:
			if file.tell() == 0:
				logger(1,'New line registered')
				self.createCSVHeaders()
				file.write(self.headers)
			data = self.createCSVData()
			logger(1,f'Write new line to {self.fileName}, data -> {data}')
			file.write(data)
		self.clearData()