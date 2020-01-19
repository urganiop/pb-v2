import pandas as pd



data = pd.read_csv('phys_prop.csv', header=0, delimiter=';')

def create_template(name, formula):
    template = f'''
    <div class="card">
    <div>{name}</div>
    <div>{formula}</div>
    </div>
    '''
    return template

def search_properties(word_list):
    cards = '''
    <div class="cards">
    '''
    for i, row in data.iterrows():
        if row['n'] in word_list:
            cards = cards + create_template(row['n'], row['f'])
    cards += '</div>'
    return cards




print(search_properties(['напряжение', 'растет', 'импульс']))

