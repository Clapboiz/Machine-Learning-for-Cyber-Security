# SAFE TEAM
# distributed under license: GPL 3 License http://www.gnu.org/licenses/
from asm_embedding.InstructionsConverter import InstructionsConverter
from asm_embedding.FunctionAnalyzerRadare import RadareFunctionAnalyzer
import json
import multiprocessing
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
import os
import random
import signal
import sqlite3
from tqdm import tqdm


class DatabaseFactory:

    def __init__(self, db_name, root_path):
        self.db_name = db_name
        self.root_path = root_path

    @staticmethod
    def worker(item):
        DatabaseFactory.analyze_file(item)
        return 0

    @staticmethod
    def extract_function(graph_analyzer):
        return graph_analyzer.extractAll()

    @staticmethod
    def insert_in_db(db_name, pool_sem, func, filename, function_name, instruction_converter):
        """ Insert the records into the tables `functions` and `filtered_functions`

        Args:
            dn_name:
            pool_sem:
            func:
            filename:
            function_name:
            instruction_converter:
        
        Returns:
            None
        """
        path = filename.split(os.sep)
        if len(path) < 4:
            return
        asm = func["asm"]
        instructions_list = func["filtered_instructions"]
        instruction_ids = json.dumps(instruction_converter.convert_to_ids(instructions_list))
        pool_sem.acquire()
        conn = sqlite3.connect(db_name)
        cur = conn.cursor()
        cur.execute('''INSERT INTO functions VALUES (?,?,?,?,?,?,?,?)''', (None,  # id
                                                                         path[-4],  # project
                                                                         path[-3],  # compiler
                                                                         path[-2],  # optimization
                                                                         path[-1],  # file_name
                                                                         function_name,  # function_name
                                                                         asm,            # asm
                                                                         len(instructions_list)) # num of instructions
                    )
        inserted_id = cur.lastrowid
        cur.execute('''INSERT INTO filtered_functions VALUES (?,?)''', (inserted_id,
                                                                        instruction_ids)
                    )
        conn.commit()
        conn.close()
        pool_sem.release()

    @staticmethod
    def analyze_file(item):
        """Read file and insert the functions information into the table `functions` and `filtered_functions`

        Returns:
            0
        """
        global pool_sem
        os.setpgrp()

        filename = item[0]
        db = item[1]
        use_symbol = item[2]
        depth = item[3]
        instruction_converter = item[4]

        analyzer =  RadareFunctionAnalyzer(filename, use_symbol, depth)
        p = ThreadPool(1)
        res = p.apply_async(analyzer.analyze)

        try:
            result = res.get(120)
        except multiprocessing.TimeoutError:
                print("Aborting due to timeout:" + str(filename))
                print('Try to modify the timeout value in DatabaseFactory instruction  result = res.get(TIMEOUT)')
                os.killpg(0, signal.SIGKILL)
        except Exception:
                print("Aborting due to error:" + str(filename))
                os.killpg(0, signal.SIGKILL)

        for func in result:
            DatabaseFactory.insert_in_db(db, pool_sem, result[func], filename, func, instruction_converter)

        analyzer.close()

        return 0

    def create_db(self):
        """ Create tables `functions` and `filtered_functions` """
        print('Database creation...')
        conn = sqlite3.connect(self.db_name)
        conn.execute(''' CREATE TABLE  IF NOT EXISTS functions (id INTEGER PRIMARY KEY, 
                                                                project text, 
                                                                compiler text, 
                                                                optimization text, 
                                                                file_name text, 
                                                                function_name text, 
                                                                asm text,
                                                                num_instructions INTEGER)
                    ''')
        conn.execute('''CREATE TABLE  IF NOT EXISTS filtered_functions  (id INTEGER PRIMARY KEY, 
                                                                         instructions_list text)
                     ''')
        conn.commit()
        conn.close()

    def scan_for_file(self, start):
        """Scan all files in `self.root_path` recursivly.

        Returns:
            An array containing all `.o` files.
        """
        file_list = []
        # Scan recursively all the subdirectory
        directories = os.listdir(start)
        for item in directories:
            item = os.path.join(start,item)
            if os.path.isdir(item):
                file_list.extend(self.scan_for_file(item + os.sep))
            elif os.path.isfile(item) and item.endswith('.o'):
                file_list.append(item)
        return file_list

    def remove_override(self, file_list):
        """Remove files that already in the table `functions` from `file_list`.

        Args:
            file_list:

        Returns:
            An array contains all files that in `file_list` but not in the table `functions`
        """
        conn = sqlite3.connect(self.db_name)
        cur = conn.cursor()
        q = cur.execute('''SELECT project, compiler, optimization, file_name FROM functions''')
        names = q.fetchall()
        names = [os.path.join(self.root_path, n[0], n[1], n[2], n[3]) for n in names]
        names = set(names)
        # If some files is already in the db remove it from the file list
        if len(names) > 0:
            print(str(len(names)) + ' Already in the database')
        cleaned_file_list = []
        for f in file_list:
            if not(f in names):
                cleaned_file_list.append(f)

        return cleaned_file_list

    # root function to create the db
    def build_db(self, use_symbol, depth):
        global pool_sem

        pool_sem = multiprocessing.BoundedSemaphore(value=1)

        instruction_converter = InstructionsConverter("D://Users//Desktop//SAFE//data//i2v//word2id.json")
        self.create_db()
        file_list = self.scan_for_file(self.root_path)

        print('Found ' + str(len(file_list)) + ' during the scan')
        file_list = self.remove_override(file_list)
        print('Find ' + str(len(file_list)) + ' files to analyze')
        random.shuffle(file_list) # why shffle?

        t_args = [(f, self.db_name, use_symbol, depth, instruction_converter) for f in file_list]

        # Start a parallel pool to analyze files
        p = Pool(processes=None, maxtasksperchild=20)
        for _ in tqdm(p.imap_unordered(DatabaseFactory.worker, t_args), total=len(file_list)):
            pass

        p.close()
        p.join()


