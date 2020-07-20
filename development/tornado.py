# -*- coding: utf-8 -*-

import argparse
import json
import logging

import tornado.escape
import tornado.httpserver
import tornado.ioloop
import tornado.options
import tornado.web

logger = logging.getLogger(__name__)


def process(text):
    return {'result': 'done'}


class Search_Handler(tornado.web.RequestHandler):
    def get(self):
        input_query = self.get_argument('query')
        result = process(input_query)
        respon_json = tornado.escape.json_encode(result)
        self.write(respon_json)

    def post(self):
        '''

        :return:
        '''
        input_query = self.request.body
        input_query_json = json.loads(input_query)
        input_query_json = input_query_json['query']
        text = input_query_json.get('text', '')
        result = process(text)
        respon_json = tornado.escape.json_encode(result)
        self.write(respon_json)


class SearchHealth_Handler(tornado.web.RequestHandler):
    # 健康检测
    def get(self):
        logger.info('search health check')
        self.write('hello world')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='An online service.')
    parser.add_argument('--port', type=int, default=8999, help='web service port, default:8999')
    args = parser.parse_args()
    app = tornado.web.Application(handlers=[(r"/search", Search_Handler),
                                            (r"/searchhealth", SearchHealth_Handler),
                                            (r"/health", SearchHealth_Handler)])

    http_server = tornado.httpserver.HTTPServer(app)

    ##多线程
    # http_server.bind(8961)
    # http_server.start(0) # Forks multiple sub-processes

    http_server.listen(args.port)  ## widows
    tornado.ioloop.IOLoop.instance().start()
