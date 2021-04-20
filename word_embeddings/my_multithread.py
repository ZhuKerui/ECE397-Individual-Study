from threading import Thread
from math import ceil

class MultiThreading:
    def __line_process_wrapper(self, line_operation, input_list:list, output_list:list):
        for line in input_list:
            result = line_operation(line)
            if result is not None:
                output_list.append(result)

    def run(self, line_operation, input_list:list, thread_num:int=1):
        if thread_num <= 0:
            return
        line_count = len(input_list)
        print('Number of lines is %d' % (line_count))
        unit_lines = ceil(line_count / thread_num)

        result = [[] for i in range(thread_num)]
        threads = [Thread(target=self.__line_process_wrapper, args=(line_operation, input_list[unit_lines*i : unit_lines*(i+1)], result[i])) for i in range(thread_num)]

        for i in range(thread_num):
            threads[i].setDaemon(True)
            threads[i].start()

        for i in range(thread_num):
            threads[i].join()
        sub_result = ['\n'.join(sub) for sub in result]
        return '\n'.join(sub_result)