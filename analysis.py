
import matplotlib.pyplot as plt
import os
import re
import codecs

# c_dir = os.getcwd()
c_dir = '/Users/XP/Documents/Src/Document Classifier/Document Classifier/'
log_dir = os.path.join(c_dir, 'log')
# capital = 'centroid'
capital = 'ridge'


def autolabel(rects, fs=5):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., height + 0.025,
                '%.2f' % float(height),
                ha='center', va='bottom', fontsize=fs)

ind = []
for i in range(20):
    ind.append(i + 0.6)

if capital == 'centroid':
    classifier_name = 'Centroid-Based Classifier'
else:
    classifier_name = 'Ridge Regression Classifier'
files = list(filter(lambda f: f.startswith(
    'log_' + capital), os.listdir(log_dir)))
files = list(filter(lambda f: f.startswith(
    'log_'), os.listdir(log_dir)))
scores = {'centroid': {'char': {}, 'word': {}},
          'ridge': {'char': {}, 'word': {}}}
for file_name in files:
    with codecs.open(os.path.join(log_dir, file_name), 'r', encoding='ascii', errors='ignore') as f:
        text = f.read()
    evaluation = text.split('Max F1 Score as a pure binary classifier\n')[
        1].split('(avg. per test object)')[0].replace(' ', '')

    elements = file_name.split('_')
    if elements[2] == 'word':
        model_name = 'Bag-of-Words Model'
        i = 1
    else:
        model_name = '5-Grams Character Model'
        i = 4
    if elements[3] == 'BINARY':
        represent_form = 'Binary Frequency'
    elif elements[3] == 'TF':
        represent_form = 'Term Frequency'
        i = i + 1
    else:
        represent_form = 'TF-IDF'
        i = i + 2

    max_f1 = []
    for r in evaluation.split('\n'):
        if len(r) == 0:
            break
        max_f1.append(float(r.split('\t')[-1]))

    s = evaluation.split('\n')[-4].split('\t')

    acc = s[1]
    pre = s[2]
    rec = s[3]
    f1 = s[4]

    scores[elements[1]][elements[2]][elements[3]] = [acc, pre, rec, f1]

    # evaluation = '''\\begin{table}[htbp]
    #     \\centering\\scriptsize
    #     \\begin{tabular}{|r|c|c|c|c|c|}
    # \\multicolumn{6}{c}{''' + classifier_name + ', ' + model_name + ', ' + represent_form + '''} \\\\
    # 				\\hline
    #                 ''' + evaluation.replace('\n\n', '\\\\ \\hline\\hline').replace(
    #     '\n', '\\\\ \\hline\n').replace('\\\\ \\hline', '\\\\\n\\hline').replace('\t', ' & ').replace('hlineM', 'hline\nM').replace('\\hline\\hlinePrecdiction', '\\hline\\hline\nPrediction ') + '''&& \\\\
    #     \\hline
    #         \\end{tabular}
    #         \\caption{Evaluation Results for ''' + classifier_name + ' using ' + model_name + ' and ' + represent_form + '''}
    #         \\label{tb:''' + classifier_name[0:2].lower() + '_' + elements[2] + '_' + elements[3].lower() + '''}
    #     \\end{table}'''
    # print(evaluation)

    col = 'b'
    # ax = plt.subplot(2, 3, i)
    # ind = []
    # if elements[1] == 'centroid':
    #     for ii in range(20):
    #         ind.append(ii + 0.6)
    #     clo = 'b'
    # else:
    #     for ii in range(20):
    #         ind.append(ii + 1)
    #     clo = 'y'

    # if i == 1 or i == 4:
    #     ax.set_ylabel(model_name, fontsize=10)
    # if i == 4:
    #     ax.set_xlabel('Binary Frequency', fontsize=10)
    # elif i == 5:
    #     ax.set_xlabel('Term Frequency', fontsize=10)
    # elif i == 6:
    #     ax.set_xlabel('TF-IDF', fontsize=10)
    # ax.set_xticks(range(1, 21))
    # ax.set_xticklabels([str(i) for i in range(1, 21)], fontsize=10)
    # # ax.set_title(classifier_name + ', ' + model_name +
    # #              ', ' + represent_form, fontsize=10)
    # ax.tick_params(axis='both', labelsize='10')
    # ax.set_xlim(0, 21)
    # ax.set_ylim(0, 1)
    # rects = ax.bar(ind, max_f1, 0.4, color=clo)
    # autolabel(rects)
    # print('''\\begin{table}[hbtp]
    #     \\centering
    #     \\scriptsize
    #     \\begin{tabular}{|r|l|}''' + text.split(
    #     'High-Weighted Tokens:\n')[1].split('\nCollecting classification solution...')[0].replace('\t', '&').replace('\n', '\\\\\n\\hline\n').replace('_', '\\_') + '''\\end{tabular}
    #         \\caption{Highest Weighted Features for ''' + classifier_name + ' using ' + model_name + ' and ' + represent_form + '''}
    #     \\end{table}''')
# plt.suptitle('Max F1 Score for ' + classifier_name)
# ax = plt.subplot(2, 3, 5)
# ax.legend(['Centroid-Based Classifier', 'Ridge Regression Classifier'], loc=2,
#           bbox_to_anchor=(-0.5, -0.2), ncol=2, fontsize=10)
# plt.suptitle('Max F1 Score Comparison')

# c = capital
# # for c in ['centroid', 'ridge']:
#
# for m in ['word', 'char']:
#     if m == 'word':
#         model_name = 'Bag-of-Words Model'
#         init = 1
#     else:
#         model_name = '5-Grams Character Model'
#         init = 5
#     for r in ['BINARY', 'TF', 'TFIDF']:
#         if r == 'BINARY':
#             represent_form = 'Binary'
#             indx = 0.8
#             col = 'b'
#         elif r == 'TF':
#             represent_form = 'Term'
#             indx = 1.8
#             col = 'r'
#         else:
#             represent_form = 'TF-IDF'
#             indx = 2.8
#             col = 'y'
#
#         for x in range(4):
#             i = init + x
#             ax = plt.subplot(2, 4, i)
#             if i == 1 or i == 5:
#                 ax.set_ylabel(model_name, fontsize=10)
#             rec = ax.bar(indx, float(scores[c][m][r][x]), 0.6, color=col)
#             ax.set_xlim(0, 4)
#             ax.set_ylim(0, 1.2)
#             ax.tick_params(axis='both', labelsize='10')
#             ax.set_xticks([])
#             autolabel(rec, 8)
#
# i = 0
# for x in ['Accuracy', 'Precision', 'Recall', 'F1']:
#     i = i + 1
#     ax = plt.subplot(2, 4, i)
#     ax.set_title(x, fontsize=10)
# ax = plt.subplot(2, 4, 6)
# ax.legend(['Binary Frequency', 'Term Frequency', 'TF-IDF'], loc=3,
#           bbox_to_anchor=(-0.5, -0.2), ncol=3, fontsize=10)
# plt.suptitle('Micro-Averaged Evaluation for ' + classifier_name)
#
# plt.show()

# confusion = """alt.atheism	22	0	0	0	0	0	0	0	0	0	0	0	0	0	2	1	0	1	0	9
#            comp.graphics	0	201	13	10	8	38	3	1	0	0	1	5	9	8	8	0	0	0	0	1
#  comp.os.ms-windows.misc	0	17	170	28	4	22	4	1	1	1	0	1	2	0	0	0	0	0	0	0
# comp.sys.ibm.pc.hardware	0	9	19	187	33	2	29	0	0	0	0	1	21	1	0	0	0	0	0	0
#    comp.sys.mac.hardware	0	12	2	16	173	1	12	1	1	1	1	0	5	2	0	0	0	0	0	0
#           comp.windows.x	0	7	8	4	1	220	1	0	0	0	0	1	2	2	1	0	0	2	0	1
#             misc.forsale	0	2	0	6	8	1	327	8	5	1	0	1	8	2	0	0	0	0	0	0
#                rec.autos	0	0	1	0	0	0	20	93	2	0	1	0	2	0	0	0	0	0	1	1
#          rec.motorcycles	0	0	1	1	1	1	3	3	65	0	1	0	1	0	0	0	0	0	0	0
#       rec.sport.baseball	0	1	0	0	0	0	1	0	1	122	1	0	0	0	0	0	0	0	0	0
#         rec.sport.hockey	0	0	0	1	0	0	4	0	1	5	148	0	1	0	1	0	0	0	0	0
#                sci.crypt	0	2	0	2	1	3	0	0	0	0	0	80	2	0	0	0	1	0	0	0
#          sci.electronics	0	7	4	15	7	2	12	9	1	2	1	2	122	6	1	0	0	0	0	0
#                  sci.med	0	5	1	0	0	0	1	0	1	1	0	1	1	116	1	3	0	0	1	1
#                sci.space	1	6	0	1	0	2	2	1	1	0	0	0	1	1	134	0	0	0	1	1
#   soc.religion.christian	5	0	0	0	0	0	0	0	0	1	0	0	0	0	1	143	0	1	2	16
#       talk.politics.guns	0	0	0	0	1	0	2	2	0	0	0	3	0	1	2	0	55	0	9	0
#    talk.politics.mideast	0	0	0	0	0	0	0	0	0	2	0	0	0	0	0	1	0	126	7	0
#       talk.politics.misc	3	1	0	1	0	1	3	2	1	1	0	3	0	5	3	1	8	3	50	5
#       talk.religion.misc	7	0	0	1	0	0	2	0	1	2	2	0	2	2	5	0	1	1	1	26"""
#
# cmap = []
# for r in confusion.split("\n"):
#     cmap.append([])
#     row = r.split("\t")
#     for c in row[1:len(row)]:
#         cmap[len(cmap) - 1].append(int(c))
#     s = sum(cmap[len(cmap) - 1])
#     cmap[len(cmap) - 1] = list(map(lambda x: x / s, cmap[len(cmap) - 1]))
#
#
# im = plt.pcolor(cmap)
# ax = plt.gca()
# ax.invert_yaxis()
# ax.set_xlabel('Actual Class', fontsize=10)
# ax.set_ylabel('Predicted Class', fontsize=10)
# ax.tick_params(axis='both', labelsize='10')
# ax.set_title(
#     'Distribution of the `Best` Classification Solution Obtained\nNormalized for each Class', fontsize=10)
# plt.colorbar(im)
# plt.show()
