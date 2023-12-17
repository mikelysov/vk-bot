import keras_nlp
from keras import losses, optimizers
import tensorflow as tf
import numpy as np

def connect_vk():
    token = 'vk1.a.-TvBOa6phzJ3a1gNHKAzP0CPkZLEulDDPopcTm_Da-V66Y7Hbp8CXzS3d90BfMFtDFZHJmoyS9fpE3Y9T4DQB_8GKCqu9oOsRSZja5REMwLvc9FlKpxnr5QR__PmVykfmDWkui1A8b8YzBtnge5iPOpvaiMf7MUBTg7eqo4ZawNaQodWJsW1L017dbcxbc1sL-xnrPpJxs_sQkLEC7ZcQA'
    authorize = vk_api.VkApi(token=token)
    longpool = VkLongPoll(authorize)
    authorize = vk_api.VkApi(token=token)
    return authorize, longpool

def write_user_msg(user_id, text):
    # запись сообщения пользователя в базу
    print(user_id, text)

def prepare():
    tokenizer = keras_nlp.models.BertTokenizer.from_preset('bert_base_multi')
    preprocessor = keras_nlp.models.BertPreprocessor.from_preset('bert_base_multi')

    model = keras_nlp.models.BertBackbone.from_preset('bert_base_multi', load_weights=True)
    model.compile(optimizer=optimizers.AdamW(5e-5), jit_compile=True)

    classifier = keras_nlp.models.XLMRobertaClassifier.from_preset('xlm_roberta_base_multi', num_classes=10, load_weights=True)
    classifier.compile(loss=losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=optimizers.AdamW(5e-5), jit_compile=True)
    
    return tokenizer, preprocessor, model, classifier

def write_user_block(user_id):
    # запись о блокировке пользователя в базу
    return True

def get_user_block_count(user_id):
    # получение из базы количество блокировок пользователя
    return 1

def set_event_user_unblock(user_id, hours):
    # установка времени для разблокировки пользователя
    return True
    
def user_block(user_id):
    # запись о блокировке пользователя в базу
    write_user_block(user_id)
    
    # получение из базы количества блокировок
    block_count = get_user_block_count(user_id)
    if block_count == 1:
        # блокировка пользователя на час
        set_event_user_unblock(user_id, 1)
    elif block_count == 2:
        # блокировка пользователя на день
        set_event_user_unblock(user_id, 24)
    #elif block_count == 3:
        # блокировка пользователя на совсем

def get_ref_by_class(klasse):
    # получение из базы ссылки по классам
    return 'http://publication.pravo.gov.ru/document/1400202312150005'

def send_msg_to_vk(authorize, user_id, msg):
    authorize.method('messages.send', {
        'user_id': user_id,
        'message': msg,
        'random_id': get_random_id()})
    return True
   
   
  authorize, longpool = connect_vk()


def main():
    _, preprocessor, model, classifier = prepare()

    for event in longpool.listen():
        if event.type == VkEventType.MESSAGE_NEW:
            user_id = event.user_id
            text = event.text
            
            # записываем сообщение в базу
            write_user_msg(user_id, text)
            
            #классификация сообщения
            predict = classifier([text], verbose=0)
            klasse = np.argmax(predict[0])
            
            if klasse == 0:
                # общение
                input_data = preprocessor([text])
                result = model(input_data)
                send_msg_to_vk(authorize, user_id, msg)
                
            elif klasse == 1:
                # блокировка пользователя
                user_block(user_id)
                
            elif klasse == 2:
                # это какой-то класс сообщения, например жкх
                input_data = preprocessor([text])
                result = model(input_data)
                # подготовили сообщение и выбрали из базы из подготовленного списка по классам нужную ссылку
                ref = get_ref_by_class(klasse)
                
                send_msg_to_vk(authorize, user_id, msg + '/n' + ref)

if __name__ == "__main__":
    main()