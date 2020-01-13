import requests
from question_to_link import qtl_main
from links_agregator import la_main
from main import ip_main


if __name__ == '__main__':
    print('starting image_parser')
    task_list = ip_main()
    answer_list = []
    for task in task_list:
        print('task: {}'.format(task))
        print('starting question_to_link')
        valid_links = qtl_main(task)
        if not valid_links:
            answer_list.append('Не нашел')
            continue
        print('valid_links', valid_links)
        print('starting links_agregator')
        answer_code = la_main(valid_links)
        print('answer_code: {}'.format(type(answer_code)))
        answer_list.append(answer_code)

    k = 1
    for answer in answer_list:
        print('<h2>Answer {}</h2><br> {}'.format(k, answer))
        k+=1