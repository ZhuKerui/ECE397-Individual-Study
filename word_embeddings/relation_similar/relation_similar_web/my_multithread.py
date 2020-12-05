import threading
from math import ceil
import signal
import io
import os
import time

is_exit = False

def tailor(output_file, input_prefix, input_posfix, num, remove=False):
    with io.open(output_file, 'w', encoding='utf-8') as dump_file:
        for i in range(num):
            with io.open(input_prefix + str(i) + input_posfix, 'r', encoding='utf-8') as load_file:
                for line in load_file:
                    if line.strip():
                        dump_file.write(line)
            if remove:
                os.remove(input_prefix + str(i) + input_posfix)

def multithread_kill(signum, frame):
    global is_exit
    is_exit = True
    print("receive a signal %d, is_exit = %d"%(signum, is_exit))

def line_process_wrapper(line_operation, freq:int, input_file:str, output_file:str, id_:int=None, start_line:int=0, lines:int=0):
    # Sanity Check
    if lines <= 0 or freq <= 0:
        return
    id_ = '' if id_ is None else str(id_)
    # Get global variable
    global is_exit
    with io.open(input_file, 'r', encoding='utf-8') as f_input:
        with io.open(output_file + id_, 'w', encoding='utf-8') as f_output:
            idx = -1
            output_buffer = []
            for idx, line in enumerate(f_input):
                if is_exit:
                    break
                if idx < start_line:
                    continue
                result = line_operation(line.strip())
                if result is not None:
                    output_buffer.append(result)
                cnt = idx - start_line + 1
                if cnt >= lines:
                    break
                if idx % freq == 0:
                    if output_buffer:
                        f_output.write(''.join(output_buffer))
                    print('Thread%s has processed %.2f' %(id_, cnt * 100 / lines))
                    output_buffer = []
            # Make sure all data is written back
            if output_buffer:
                f_output.write(''.join(output_buffer))
            if is_exit:
                print('Thread%s is terminated' % (id_))
            else:
                print('Extract context accomplished with %d lines processed' % (1 + idx - start_line))

def multithread_wrapper(line_operation, freq:int, input_file:str, output_file:str, thread_num:int=1):
    global is_exit
    
    if thread_num <= 0:
        return
    # Count the number of lines
    line_count = -1
    with io.open(input_file, 'r', encoding='utf-8') as f_load:
        for line_count, line in enumerate(f_load):
            pass
        line_count += 1
    print('Number of lines is %d' % (line_count))

    unit_lines = ceil(line_count / thread_num)
    threads = []
    signal.signal(signal.SIGINT, multithread_kill)
    signal.signal(signal.SIGTERM, multithread_kill)
    is_exit = False
    for i in range(thread_num):
        id_ = None if thread_num == 1 else i
        start_line = unit_lines * i
        t = threading.Thread(target=line_process_wrapper, args=(line_operation, freq, input_file, output_file, id_, start_line, unit_lines))
        # t.setDaemon(True)
        threads.append(t)
    for i in range(thread_num):
        threads[i].start()
    while 1:
        alive = False
        for i in range(thread_num):
            alive = alive or threads[i].isAlive()
        if not alive:
            break

    if thread_num > 1:
        tailor(output_file, output_file, '', thread_num, remove=True)