# from graph.graph_construction.session_graph import define_sessions, create_weightless_session_graph
# list = [1, 2, 3, 4, 5, 6, 7]
# for start in range(0, len(list)):
#     print(start)
# print(len(list))


cells = [10, 20, 50]
for i, c in enumerate(cells):
    if i == 0:
        print(f"first: {c}")
    # elif i <
    else:
        # print(1)
        print(f"hidden: {c}" if i != len(cells) - 1 else f"out: {c}")
