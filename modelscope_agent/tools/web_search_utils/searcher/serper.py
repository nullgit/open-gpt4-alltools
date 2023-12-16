import json
import os

import requests
from modelscope_agent.tools.web_search_utils.search_util import (
    AuthenticationKey, SearchResult)
from modelscope_agent.tools.web_search_utils.searcher.base_searcher import \
    WebSearcher


class SerperWebSearcher(WebSearcher):

    def __init__(
        self,
        timeout=10000,
        mkt='en-US',
        endpoint='https://google.serper.dev/search',
        **kwargs,
    ):
        self.mkt = mkt
        self.endpoint = endpoint
        self.timeout = timeout
        self.token = os.environ.get('SERPER_API_KEY')

    def __call__(self, query, **kwargs):

        params = {
            "q": query,
            "gl": "cn",  # 中国
            "hl": "zh-cn"  # 简体中文
        }

        headers = {
            'X-API-KEY': self.token,
            'Content-Type': 'application/json'
        }

        if kwargs:
            params.update(kwargs)
        try:
            response = requests.post(
                self.endpoint,
                headers=headers,
                params=params,
                timeout=self.timeout)
            print('搜索结果', response.text)
            raw_result = json.loads(response.text)
            if raw_result.get('error', None):
                print(f'Call Serper web search api failed: {raw_result}')
        except Exception as ex:
            raise ex('Call Serper web search api failed.')

        results = []
        res_list = raw_result.get('organic', [])
        for item in res_list:
            title = item.get('title', None)
            link = item.get('link', None)
            sniper = item.get('snippet', None)
            if not link and not sniper:
                continue

            results.append(SearchResult(title=title, link=link, sniper=sniper))

        return results


if __name__ == '__main__':

    searcher = SerperWebSearcher()
    res = searcher('哈尔滨元旦的天气情况')
    print([item.__dict__ for item in res])
