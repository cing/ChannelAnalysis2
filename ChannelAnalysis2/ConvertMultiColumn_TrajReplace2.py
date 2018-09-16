# To support pandas dataframes, this script converts the variable column length data of the coordination data into a format
# better suited for easily timeseries generation. This is mainly geared for the coordination input data where each datapoint is
# the position of an ion at a given time.
from sys import argv
import gzip

#a great helper function to iterate over chunks of a list
def chunker(seq, size):
    return (seq[pos:pos + size] for pos in xrange(0, len(seq), size))

# A helper function to open with gzip if the file is gzipped
def file_opener(fname):
    if fname.endswith('.gz'):
        return gzip.open(fname)
    else:
        return open(fname)

# This function does the legwork, of opening the files and splitting each line into chunks
# of length equal to cols_per_entry. This returns a list of lists.
def parse_legacy_datatype(fnames, header_cols=0, cols_per_entry=12):
    return_list = []

    for fname in fnames:
        with file_opener(fname) as data_raw:
            data_raw_split = [line.strip().split() for line in data_raw.readlines()]
            for sline in data_raw_split: 
                #print sline
                for chunk in chunker(sline, cols_per_entry):
                    #print chunk
                    tchunk=list(chunk)
                    # Change the second last line
                    #tchunk[-2] = str(float(tchunk[-2])+10)
                    # Change the second line
                    tchunk[1] = str(float(tchunk[1])+15)
                    return_list.append(tchunk)

                return_list.append("\n")
    return return_list

if __name__ == '__main__':
    #for line in parse_legacy_datatype(argv[1:]):
    for line in parse_legacy_datatype(argv[1:], cols_per_entry=14):
    #for line in parse_legacy_datatype(argv[1:], cols_per_entry=12):
        print " ".join(line),

    #print parse_legacy_datatype(argv[1:])

