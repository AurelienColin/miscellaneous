def flatten(nested_list):
    if not isinstance(nested_list[0], list):
        return nested_list

    flat_list = [e for child_list in nested_list for e in flatten(child_list)]
    return flat_list


def remove_duplicates(l):
    return list(dict.fromkeys(l))


def prune(l):
    while len(l) != 0 and l[0] == 0:
        l = l[1:]
    return l