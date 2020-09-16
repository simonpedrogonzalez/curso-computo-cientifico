def ambos(s):
    if len(s) > 2:
        return s[:2]+s[-2:]
    else:
        return ''


def fix(s):
    for c in s:
        s = '*'.join(s.rsplit(c, s.count(c)-1))
    return s


def mezclar(a, b):
    return b[0]+a[1:]+' '+a[0]+b[1:]


def macheos(lista):
    return sum([len(s) >= 2 and s[-1] == s[-2] for s in lista])


def front_x(lista):
    x, no_x = [], []
    for s in lista:
        x.append(s) if s[0] == 'x' else no_x.append(s)
    x.sort()
    no_x.sort()
    return x + no_x


def sort_last(lista):
    lista.sort(key=lambda x: x[-1])
    return lista


def tablas(nro):
    return [nro*i for i in range(1, 11)]


def mapeo(s):
    return {c: i for i, c in enumerate(set(s))}


def busqueda_reversa(d, n):
    return {v: k for k, v in d.items()}.get(n)


def invitados(d):
    return [k for k, v in d.items() if v == 'Asistirá']


def justificar(text, chars_per_line):
    lista = text.split()
    i = 1
    lines = []
    current_line = lista[0]
    while i < len(lista):
        if len(current_line) + len(lista[i]) + 1 <= chars_per_line:
            current_line += ' ' + lista[i]
        else:
            lines.append(current_line +
                         ' ' * (chars_per_line - len(current_line))
                         + '\r\n')
            current_line = lista[i]
        i += 1
    lines.append(current_line+' '*(chars_per_line-len(current_line)))
    return ''.join(lines)


def main():
    print(fix('palabra'))
    print(mezclar('mix', 'pod'))
    print(macheos(['a', 'aa', 'baa', 'bab']))
    print(front_x(['mix', 'xyz', 'apple', 'xanadu', 'aardvark']))
    print(sort_last([(1, 7), (1, 3), (3, 4, 5), (2, 2)]))
    print(tablas(3))
    print(mapeo("casa"))
    print(busqueda_reversa(mapeo("casa"), 2))
    print(invitados({'a': 'Asistirá',
                     'b': 'No asistirá',
                     'c': 'No asistirá',
                     'd': 'Asistirá'}))
    texto = "Este es el texto a justificar y es muy largo y que se yo"
    print(justificar(texto, 20))


if __name__ == "__main__":
    main()
