"""This script downloads the necessary files for bioformats unittests
source: http://loci.wisc.edu/software/sample-data"""

def download_bioformats_tstfiles():
    import os
    try:
        import jpype
    except ImportError:
        raise ImportError("JPype is required for running tests on these files.")
    from urllib.request import urlretrieve
    from zipfile import ZipFile

    path, _ = os.path.split(os.path.abspath(__file__))
    path = os.path.join(path, 'pims', 'tests', 'data', 'bioformats')

    def get_bioformats_file(filename, filepath='', url=''):
        if url == '':
            url = 'https://samples.scif.io/' + filename
        fn = os.path.join(filepath, filename)
        urlretrieve(url, fn)
        try:
            with ZipFile(fn) as zf:
                zf.extractall(filepath)
            os.remove(fn)
            if os.path.isfile('readme.txt'):
                os.remove('readme.txt')
        except:
            print("Failed to extract {}".format(fn))
            raise

    passing = ['qdna1.zip', 'leica_stack.zip', 'Blend_Final.zip', 'HEART.zip',
               'wtembryo.zip', 'mitosis-test.zip', 'dnasample1.zip',
               '2chZT.zip', 'mouse-kidney.zip', 'MF-2CH-Z-T.zip',
               '10-31_E1.zip', 'KEVIN2-3.zip']

    #failing = ['sdub.zip', 'NESb.zip', 'TAABA.zip', 'embryo2.zip', 'dub.zip']

    for fn in passing:
        get_bioformats_file(fn, path)
        print('Downloaded {}'.format(fn))

if __name__ == '__main__':
    download_bioformats_tstfiles()
