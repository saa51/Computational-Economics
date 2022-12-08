def write_markdown_table(content, title=None, index=None, align='l'):
    output = ''
    row_num = len(content)
    col_num = len(content[0])

    def __write_markdown_table_line(content_line):
        line = '| '
        for c in content_line:
            line += str(c) + ' | '
        line += '\n'
        return line

    if title is None:
        title = ['column ' + str(i) for i in range(col_num)]
    if index is not None:
        title = ['index'] + title
        col_num += 1
    output += __write_markdown_table_line(title)

    if align == 'r':
        pattern = '----:'
    elif align == 'c':
        pattern = ':----:'
    else:
        pattern = ':----'
    output += __write_markdown_table_line([pattern for _ in range(col_num)])
    for l in range(row_num):
        output += __write_markdown_table_line([index[l]] + content[l])
    return output
