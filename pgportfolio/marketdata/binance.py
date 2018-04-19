import json
import time
import math
import sys
from datetime import datetime
import pgportfolio.constants as const

if sys.version_info[0] == 3:
    from urllib.request import Request, urlopen
    from urllib.parse import urlencode
else:
    from urllib2 import Request, urlopen
    from urllib import urlencode


# Possible Commands
PUBLIC_COMMANDS = ['api/v1/ticker/24hr', 'exchange/public/product', 'api/v1/klines']


class Binance:
    def __init__(self, APIKey='', Secret=''):
        self.APIKey = APIKey.encode()
        self.Secret = Secret.encode()
        # Conversions
        self.timestamp_str = lambda timestamp=time.time(), format="%Y-%m-%d %H:%M:%S":\
            datetime.fromtimestamp(timestamp).strftime(format)
        self.str_timestamp = lambda datestr=self.timestamp_str(), format="%Y-%m-%d %H:%M:%S":\
            int(time.mktime(time.strptime(datestr, format)))
        self.float_roundPercent = lambda floatN, decimalP=2:\
            str(round(float(floatN) * 100, decimalP)) + "%"

        # PUBLIC COMMANDS
        self.marketTicker = lambda x=0: self.api('api/v1/ticker/24hr', 'marketTicker')
        self.marketVolume = lambda x=0: self.api('exchange/public/product', 'marketVolume')
        self.marketStatus = lambda x=0: self.api('exchange/public/product', 'marketStatus')
        # self.marketLoans = lambda coin: self.api('returnLoanOrders',{'currency':coin})
        # self.marketOrders = lambda pair='all', depth=10:\
        #     self.api('returnOrderBook', {'currencyPair':pair, 'depth':depth})
        self.marketChart = lambda pair, period=const.DAY, start=time.time() - (const.WEEK * 1), end=time.time():\
            self.api('api/v1/klines', 'marketChart', {'symbol': self.convertToSymbol(pair),
                                                      'interval': self.convertPeriodToInterval(period),
                                                      'startTime': int(start * 1000),
                                                      'endTime': int(end * 1000),
                                                      'limit': int((end - start) / period)})
        # self.marketTradeHist = lambda pair: self.api('returnTradeHistory',{'currencyPair':pair})

    #####################
    # Main Api Function #
    #####################
    def api(self, command, convertionType, args={}):
        """
        returns 'False' if invalid command or if no APIKey or Secret is specified (if command is "private")
        returns {"error":"<error message>"} if API error
        """
        if command == 'api/v1/klines' and args['limit'] > 500:
            result = []
            # break request down in to smaller requests
            nrrequests = math.ceil(args['limit'] / 500)
            for i in range(nrrequests):
                arg = args.copy()
                arg['limit'] = 500
                arg['startTime'] = args['startTime'] + self.convertIntervalToPeriod(args['interval']) * 500 * i * 1000
                arg['endTime'] = args['startTime'] + self.convertIntervalToPeriod(args['interval']) * 500 * (i + 1) * 1000
                result = result + self.api(command, convertionType, arg)
                time.sleep(.100)  # prevent rate limit issues (max 20 r/s)
            return sorted([dict(t) for t in set([tuple(d.items()) for d in result])], key=lambda k: k['date'])

        elif command in PUBLIC_COMMANDS:
            url = 'https://www.binance.com/'
            url += command
            ret = urlopen(Request(url + '?' + urlencode(args)))
            return self.convertToPoloniexFormat(convertionType, json.loads(ret.read().decode(encoding='UTF-8')))
        else:
            return False

    def convertPeriodToInterval(self, period):
        choices = {
            const.MINUTE: '1m',
            const.THREE_MINUTE: '3m',
            const.FIVE_MINUTE: '5m',
            const.FIFTEEN_MINUTE: '15m',
            const.HALF_HOUR: '30m',
            const.HOUR: '1h',
            const.TWO_HOUR: '2h',
            const.FOUR_HOUR: '4h',
            const.SIX_HOUR: "6h",
            const.EIGHT_HOUR: '8h',
            const.HALF_DAY: '12h',
            const.DAY: '1d',
            const.THREE_DAY: '3d',
            const.WEEK: '1w',
            const.MONTH: '1M',
        }
        return choices.get(period, '1d')

    def convertIntervalToPeriod(self, period):
        choices = {
            '1m': const.MINUTE,
            '3m': const.THREE_MINUTE,
            '5m': const.FIVE_MINUTE,
            '15m': const.FIFTEEN_MINUTE,
            '30m': const.HALF_HOUR,
            '1h': const.HOUR,
            '2h': const.TWO_HOUR,
            '4h': const.FOUR_HOUR,
            '6h': const.SIX_HOUR,
            '8h': const.EIGHT_HOUR,
            '12h': const.HALF_DAY,
            '1d': const.DAY,
            '3d': const.THREE_DAY,
            '1w': const.WEEK,
            '1M': const.MONTH,
        }
        return choices.get(period, 86400)

    def convertToSymbol(self, pair):
        coin1, coin2 = pair.split('_')
        return coin2 + coin1

    def convertFromSymbol(self, symbol):
        index = symbol.find('BTC')
        if index == 0:
            index = 3
        return '_'.join((symbol[index:], symbol[:index],))

    def convertToPoloniexFormat(self, convertionType, response):
        if convertionType == 'marketTicker':
            result = {}
            for item in response:
                if 'BTC' not in item['symbol']:
                    continue
                if 'BNB' in item['symbol']:
                    continue
                pair = self.convertFromSymbol(item['symbol'])
                result[pair] = {
                    'last': item['lastPrice'],
                    'lowestAsk': item['askPrice'],
                    'highestBid': item['bidPrice'],
                    'percentChange': item['priceChangePercent'],
                    'baseVolume': item['quoteVolume'],  # reverse as Binance takes BTC as first currency pair
                    'quoteVolume': item['volume'],
                    'isFrozen': 0,
                    'high24hr': item['highPrice'],
                    'low24hr': item['lowPrice']
                }
            return result
        elif convertionType == 'marketVolume':
            result = {}
            for item in response['data']:
                if 'BTC' not in item['symbol']:
                    continue
                if 'BNB' in item['symbol']:
                    continue
                pair = self.convertFromSymbol(item['symbol'])
                coin1, coin2 = pair.split('_')
                result[pair] = {
                    coin1: round(float(item['volume']) * float(item['close']), 6),  # BTC
                    coin2: round(float(item['volume']), 6)
                }
            return result
        elif convertionType == 'marketStatus':
            result = {}
            for item in response['data']:
                if 'BTC' not in item['symbol']:
                    continue
                if 'BNB' in item['symbol']:
                    continue
                pair = self.convertFromSymbol(item['symbol'])
                result[pair.split('_')[1]] = {
                    'name': item['baseAssetName'],
                    'txFee': 0.0005,
                    'disabled': 0 if item['active'] else 1,
                    'frozen': 1 if item['status'] != "TRADING" else 0
                }
            return result

        elif convertionType == 'marketChart':
            result = []
            for item in response:
                result.append({
                    'date': int(item[0] / 1000),
                    'high': float(item[2]),
                    'low': float(item[3]),
                    'open': float(item[1]),
                    'close': float(item[4]),
                    'volume': float(item[7]),
                    'quoteVolume': float(item[5]),
                    'weightedAverage': 0
                })
            return result
        else:
            return response
