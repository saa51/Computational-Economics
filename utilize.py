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
        if index is not None:
            output += __write_markdown_table_line([index[l]] + content[l])
        else:
            output += __write_markdown_table_line(content[l])
    return output


def write_latex_table(content, title=None, index=None, align='c', hline=False, vline=False):
    output = '\\begin{tabular}'
    row_num = len(content)
    col_num = len(content[0])
    if index is not None:
        col_num += 1
    format_str = '{|' + (align + ('|' if vline else '')) * col_num
    if not vline:
        format_str += '|'
    output += format_str + '}\n\\hline\n'

    def __write_latex_table_line(content_line, hline):
        line = ''
        for c in content_line[:-1]:
            line += str(c) + ' & '
        line += str(content_line[-1])
        line += '\\\\\n'
        if hline:
            line += '\\hline \n'
        return line

    if title is not None:
        output += __write_latex_table_line(['index'] + title, hline)

    for l in range(row_num):
        if index is None:
            output += __write_latex_table_line(content[l], hline)
        else:
            output += __write_latex_table_line([index[l]] + content[l], hline)

    if not hline:
        output += '\\hline\n'
    output += '\\end{tabular}\n'
    return output