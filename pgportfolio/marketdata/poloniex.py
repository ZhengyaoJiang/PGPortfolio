import json
import time
import sys
from datetime import datetime

if sys.version_info[0] == 3:
    from urllib.request import Request, urlopen
    from urllib.parse import urlencode
else:
    from urllib2 import Request, urlopen
    from urllib import urlencode

minute = 60
hour = minute*60
day = hour*24
week = day*7
month = day*30
year = day*365

# Possible Commands
PUBLIC_COMMANDS = ['returnTicker', 'return24hVolume', 'returnOrderBook', 'returnTradeHistory', 'returnChartData', 'returnCurrencies', 'returnLoanOrders']

class Poloniex:
    def __init__(self, APIKey='', Secret=''):
        self.APIKey = APIKey.encode()
        self.Secret = Secret.encode()
        # Conversions
        self.timestamp_str = lambda timestamp=time.time(), format="%Y-%m-%d %H:%M:%S": datetime.fromtimestamp(timestamp).strftime(format)
        self.str_timestamp = lambda datestr=self.timestamp_str(), format="%Y-%m-%d %H:%M:%S": int(time.mktime(time.strptime(datestr, format)))
        self.float_roundPercent = lambda floatN, decimalP=2: str(round(float(floatN) * 100, decimalP))+"%"

        # PUBLIC COMMANDS
        self.marketTicker = lambda x=0: self.api('returnTicker')
        self.marketVolume = lambda x=0: self.api('return24hVolume')
        self.marketStatus = lambda x=0: self.api('returnCurrencies')
        self.marketLoans = lambda coin: self.api('returnLoanOrders',{'currency':coin})
        self.marketOrders = lambda pair='all', depth=10:\
            self.api('returnOrderBook', {'currencyPair':pair, 'depth':depth})
        self.marketChart = lambda pair, period=day, start=time.time()-(week*1), end=time.time(): self.api('returnChartData', {'currencyPair':pair, 'period':period, 'start':start, 'end':end})
        self.marketTradeHist = lambda pair: self.api('returnTradeHistory',{'currencyPair':pair}) # NEEDS TO BE FIXED ON Poloniex

    #####################
    # Main Api Function #
    #####################
    def api(self, command, args={}):
        """
        returns 'False' if invalid command or if no APIKey or Secret is specified (if command is "private")
        returns {"error":"<error message>"} if API error
        """
        if command in PUBLIC_COMMANDS:
            url = 'https://poloniex.com/public?'
            args['command'] = command
            ret = urlopen(Request(url + urlencode(args)))
            return json.loads(ret.read().decode(encoding='UTF-8'))
        else:
            return False
