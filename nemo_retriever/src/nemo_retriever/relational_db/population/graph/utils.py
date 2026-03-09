import regex


def flat_list_recursive(nested_list):
    output = []
    for i in nested_list:
        if isinstance(i, list):
            temp = flat_list_recursive(i)
            for j in temp:
                output.append(j)
        else:
            output.append(i)
    return output


def remove_redundant_parentheses(text):
    r = "s/(\(|^)\K(\((((?2)|[^()])*)\))(?=\)|$)/\\3/"
    if r[0] != "s":
        raise SyntaxError('Missing "s"')
    d = r[1]
    r = r.split(d)
    if len(r) != 4:
        raise SyntaxError("Wrong number of delimiters")
    flags = 0
    count = 1
    for f in r[3]:
        if f == "g":
            count = 0
        else:
            flags |= {
                "i": regex.IGNORECASE,
                "m": regex.MULTILINE,
                "s": regex.DOTALL,
                "x": regex.VERBOSE,
            }[f]
    s = r[2]
    r = r[1]
    # z = 0

    while 1:
        m = regex.subn(r, s, text, count, flags)
        text = m[0]
        if m[1] == 0:
            break

    return text


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]
