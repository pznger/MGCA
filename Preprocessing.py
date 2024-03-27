from Hypers import *
from Utils import *
from nltk.tokenize import word_tokenize
import json
import numpy as np


def read_news(data_root_path):
    print('preprocessing-read_news')
    # 新闻
    news = {}

    news_index = {}
    index = 1

    word_dict = {}
    word_index = 1

    content_count = {}
    content_dict = {}
    content_index = 1

    entity_dict = {}
    entity_index = 1

    category_dict = {}
    category_index = 1

    subcategory_dict = {}
    subcategory_index = 1

    for path in ['train', 'dev']:
        print('path:', os.path.join(data_root_path, path, 'news.tsv'))

        with open(os.path.join(data_root_path, path, 'news.tsv'), encoding='utf-8') as f:
            lines = f.readlines()  # 101527

        for line in lines:
            splited = line.strip('\n').split('\t')
            doc_id, vert, subvert, title, abstract, url, entity, _ = splited
            # 文档id,主题，副主题，标题，摘要，路径，实体
            entity = json.loads(entity)  # json.loads()将str类型的数据转换为dict类型
            if doc_id in news_index:
                continue

            news_index[doc_id] = index
            index += 1

            title = word_tokenize(title.lower())  # 将标题切分为单词
            abstract = abstract.lower().split()[:MAX_CONTENT]  # 将标题切分为单词，最多50个
            entity = [e['WikidataId'] for e in entity]

            news[doc_id] = [vert, subvert, title, entity, abstract]

            for word in title:
                if not (word in word_dict):
                    word_dict[word] = word_index
                    word_index += 1

            for word in abstract:
                if not (word in content_count):
                    content_count[word] = 0
                content_count[word] += 1

            for e in entity:
                if not (e in entity_dict):
                    entity_dict[e] = entity_index
                    entity_index += 1

            if not vert in category_dict:
                category_dict[vert] = category_index
                category_index += 1

            if not subvert in subcategory_dict:
                subcategory_dict[subvert] = subcategory_index
                subcategory_index += 1

    for word in content_count:
        if content_count[word] < 3:
            continue
        content_dict[word] = content_index
        content_index += 1

    return news, news_index, category_dict, subcategory_dict, word_dict, content_dict, entity_dict


def get_doc_input(news, news_index, category_dict, subcategory_dict, word_dict, content_dict, entity_dict):
    news_num = len(news) + 1
    news_title = np.zeros((news_num, MAX_TITLE), dtype='int32')
    news_vert = np.zeros((news_num,), dtype='int32')
    news_subvert = np.zeros((news_num,), dtype='int32')
    news_entity = np.zeros((news_num, MAX_ENTITY), dtype='int32')
    news_content = np.zeros((news_num, MAX_CONTENT), dtype='int32')

    for key in news:
        vert, subvert, title, entity, content = news[key]
        doc_index = news_index[key]

        news_vert[doc_index] = category_dict[vert]
        news_subvert[doc_index] = subcategory_dict[subvert]

        for word_id in range(min(MAX_TITLE, len(title))):
            news_title[doc_index, word_id] = word_dict[title[word_id]]

        for entity_id in range(min(MAX_ENTITY, len(entity))):
            news_entity[doc_index, entity_id] = entity_dict[entity[entity_id]]

        for content_id in range(min(MAX_ENTITY, len(content))):
            if not content[content_id] in content_dict:
                continue
            news_content[doc_index, content_id] = content_dict[content[content_id]]

    return news_title, news_vert, news_subvert, news_entity, news_content


def load_entity_metadata(KG_root_path):
    print('load_entity_metadata')
    # Entity Table
    with open(os.path.join(KG_root_path, 'entity2id.txt')) as f:
        lines = f.readlines()

    EntityId2Index = {}
    EntityIndex2Id = {}
    for i in range(1, len(lines)):
        eid, eindex = lines[i].strip('\n').split('\t')
        EntityId2Index[eid] = int(eindex)
        EntityIndex2Id[int(eindex)] = eid

    entity_embedding = np.load(os.path.join(KG_root_path, 'entity_embedding.npy'))
    entity_embedding = np.concatenate([entity_embedding, np.zeros((1, 100))], axis=0)

    with open(os.path.join(KG_root_path, 'KGGraph.json')) as f:
        s = f.read()
    graph = json.loads(s)

    return graph, EntityId2Index, EntityIndex2Id, entity_embedding


def load_news_entity(news, EntityId2Index, data_root_path):
    print('load_news_entity')
    with open(os.path.join(data_root_path, 'docs.tsv'), encoding='utf-8') as f:
        lines = f.readlines()
    news_entity = {}
    for i in range(len(lines)):
        docid, _, _, _, _, _, entities, _ = lines[i].strip('\n').split('\t')
        entities = json.loads(entities)
        news_entity[docid] = []
        for j in range(len(entities)):
            e = entities[j]['Label']
            eid = entities[j]['WikidataId']
            if not eid in EntityId2Index:
                continue
            news_entity[docid].append([e, eid, EntityId2Index[eid]])

    return news_entity


def parse_zero_hop_entity(EntityId2Index, news_entity, news_index, max_entity_num=5):
    news_entity_index = np.zeros((len(news_index) + 1, max_entity_num), dtype='int32') + len(EntityId2Index)
    for newsid in news_index:
        index = news_index[newsid]
        entities = news_entity[newsid]
        ri = np.random.permutation(len(entities))
        for j in range(min(len(entities), max_entity_num)):
            e = entities[ri[j]][-1]
            news_entity_index[index, j] = e
    return news_entity_index


def parse_one_hop_entity(EntityId2Index, EntityIndex2Id, news_entity_index, graph, news_index, max_entity_num=5):
    one_hop_entity = np.zeros((len(news_index) + 1, max_entity_num, max_entity_num), dtype='int32') + len(
        EntityId2Index)
    for newsid in news_index:
        index = news_index[newsid]
        entities = news_entity_index[index]
        for j in range(max_entity_num):
            eindex = news_entity_index[index, j]
            if eindex == len(EntityId2Index):
                continue
            eid = EntityIndex2Id[eindex]
            neighbors = graph[eid]
            rindex = np.random.permutation(len(neighbors))
            for k in range(min(max_entity_num, len(neighbors))):
                nindex = rindex[k]
                neig_id = neighbors[nindex]
                # print(neig_id)
                neig_index = EntityId2Index[neig_id]
                one_hop_entity[index, j, k] = neig_index
    return one_hop_entity


def read_train_clickhistory(news_index, data_root_path, filename):
    print('read_train_clickhistory()')

    lines = []
    with open(os.path.join(data_root_path, filename)) as f:
        lines = f.readlines()  # lines:{list:2232748}

    sessions = []
    for i in range(len(lines)):
        _, uid, eventime, click, imps = lines[i].strip().split('\t')
        # _ :impression编号  uid：用户id  时间  impression之前的点击新闻历史序列  这个印象中展示的新闻列表
        if click == '':
            clikcs = []
        else:
            clikcs = click.split()
        true_click = []  # 切割点击历史
        for click in clikcs:
            if not click in news_index:
                continue
            true_click.append(click)
        pos = []
        neg = []
        for imp in imps.split():
            docid, label = imp.split('-')  # 将新闻id与印象切割
            if label == '1':
                pos.append(docid)  # 用户点击新闻集合
            else:
                neg.append(docid)  # 用户未点击新闻集合
        sessions.append([true_click, pos, neg])
    sessions = np.array(sessions)
    # np.save('/data/linxinze/test/train_sessions.npy', sessions)
    return sessions


def read_test_clickhistory(news_index, data_root_path, filename):
    lines = []
    with open(os.path.join(data_root_path, filename)) as f:
        for i in range(200000):
            l = f.readline()
            lines.append(l)

    sessions = []
    for i in range(len(lines)):
        _, uid, eventime, click, imps = lines[i].strip().split('\t')
        if click == '':
            clikcs = []
        else:
            clikcs = click.split()
        true_click = []
        for click in clikcs:
            if not click in news_index:
                continue
            true_click.append(click)
        pos = []
        neg = []
        for imp in imps.split():
            docid, label = imp.split('-')
            if label == '1':
                pos.append(docid)
            else:
                neg.append(docid)
        sessions.append([true_click, pos, neg])
    # np.save('/data/linxinze/test/test_sessions.npy', sessions)
    return sessions


def read_test_clickhistory_noclk(news_index, data_root_path, filename):
    lines = []
    with open(os.path.join(data_root_path, filename)) as f:
        lines = f.readlines()
    sessions = []
    for i in range(len(lines)):  # len(lines)=73152
        _, uid, eventime, click, imps = lines[i].strip().split('\t')
        if click == '':
            clicks = []
        else:
            clicks = click.split()
        true_click = []  # 验证集用户历史点击新闻
        for j in range(len(clicks)):
            click = clicks[j]
            assert click in news_index
            true_click.append(click)
        pos = []
        neg = []
        for imp in imps.split():
            docid, label = imp.split('-')
            if label == '1':
                pos.append(docid)
            else:
                neg.append(docid)
        sessions.append([true_click, pos, neg])
    return sessions


# 构建训练数据 用户
def parse_user(news_index, session):
    # len(session)=156965
    user_num = len(session)
    # user:{'click':array(156965,50)}
    user = {'click': np.zeros((user_num, MAX_CLICK), dtype='int32'), }
    for user_id in range(len(session)):
        tclick = []
        click, pos, neg = session[user_id]
        for i in range(len(click)):
            tclick.append(news_index[click[i]])
        click = tclick  # 用户之前的点击历史

        if len(click) > MAX_CLICK:
            click = click[-MAX_CLICK:]
        else:
            click = [0] * (MAX_CLICK - len(click)) + click  # 前面全是0，最后是用户之前点击新闻的编号

        user['click'][user_id] = np.array(click)
    return user


def get_train_input(news_index, session):
    print('get_train_input()')
    sess_pos = []  # 正样本
    sess_neg = []  # 负样本1：4
    user_id = []  # 用户id
    for sess_id in range(len(session)):
        sess = session[sess_id]
        _, poss, negs = sess
        for i in range(len(poss)):
            pos = poss[i]
            neg = newsample(negs, npratio)
            sess_pos.append(pos)
            sess_neg.append(neg)
            user_id.append(sess_id)
    # sess_pos sess_neg user_id {list:236344}

    sess_all = np.zeros((len(sess_pos), 1 + npratio), dtype='int32')  # (236344,5)
    label = np.zeros((len(sess_pos), 1 + npratio))  # (236344,5)
    for sess_id in range(sess_all.shape[0]):  # sess_all.shape[0]=236344
        pos = sess_pos[sess_id]  # 正样本新闻
        negs = sess_neg[sess_id]  # 负样本新闻 4个
        sess_all[sess_id, 0] = news_index[pos]
        index = 1
        for neg in negs:
            sess_all[sess_id, index] = news_index[neg]
            index += 1
        label[sess_id, 0] = 1
    user_id = np.array(user_id, dtype='int32')

    return sess_all, user_id, label


def get_test_input(news_index, session):
    # session{list:73152}
    Impressions = []
    userid = []
    UserIds = []
    DocIds = []
    Labels = []
    Bound = []
    count = 0
    for sess_id in range(len(session)):
        _, poss, negs = session[sess_id]
        imp = {'labels': [],
               'docs': []}

        start = count
        for i in range(len(poss)):
            docid = news_index[poss[i]]
            imp['docs'].append(docid)
            imp['labels'].append(1)
            DocIds.append(docid)
            Labels.append(1)
            userid.append(sess_id)
            # userid.append(sess_id)
            count += 1
        for i in range(len(negs)):
            docid = news_index[negs[i]]
            imp['docs'].append(docid)
            imp['labels'].append(0)
            DocIds.append(docid)
            Labels.append(0)
            userid.append(sess_id)
            count += 1
        Bound.append([start, count])
        # for i in range(len(negs)):
        #     docid = news_index[negs[i]]
        #     imp['docs'].append(docid)
        #     imp['labels'].append(0)
        Impressions.append(imp)

    DocIds = np.array(DocIds, dtype='int32')
    UserIds = np.array(userid, dtype='int32')
    Labels = np.array(Labels, dtype='float32')

    return DocIds, UserIds, Labels, Bound
