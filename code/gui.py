import tkinter as tk
from tkinter import ttk
import threading
import queue
import config
from main_ag import run_ag
from mcmc import run_mcmc
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def show_plot_window(self, plot_data):
        # Tworzy nowe okienko zintegrowane z Tkinterem
        plot_win = tk.Toplevel(self.root)
        plot_win.title("Wykres Ewolucji Algorytmu")
        plot_win.geometry("800x600")

        # Rysowanie za pomocą obiektowego API Matplotliba
        fig = Figure(figsize=(10,6), dpi=100)
        ax = fig.add_subplot(111)

        generations_range = plot_data['generations_range']
        fitnessHistory = plot_data['fitnessHistory']
        avgHistory = np.array(plot_data['avgHistory'])
        stdHistory = np.array(plot_data['stdHistory'])
        medHistory = plot_data['medHistory']
        mutatePoints = plot_data['mutatePoints']
        trueFitness = plot_data['trueFitness']

        ax.fill_between(generations_range, avgHistory - stdHistory, avgHistory + stdHistory, color='gray', alpha=0.2, label='Odchylenie std')
        ax.plot(generations_range, fitnessHistory, color='green', linewidth=2, label='best fitness')
        ax.plot(generations_range, medHistory, color='blue', linewidth=1, label='mediana z populacji')
        ax.set_title('Proces uczenia Algorytmu Genetycznego', fontsize=14)
        ax.set_xlabel('Pokolenie (Iteration)', fontsize=12)
        ax.set_ylabel('Fitness Score', fontsize=12)
        ax.axhline(y=trueFitness, color='r', linestyle='--', label='Original fitness')
        
        if mutatePoints:
            ax.vlines(mutatePoints, ax.get_ylim()[0], ax.get_ylim()[1], colors='orange', linestyles='--', alpha=0.4, label='Mutacja wymuszona')
            
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()

        # Osadzenie wykresu w oknie Tkintera
        canvas = FigureCanvasTkAgg(fig, master=plot_win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

class CryptanalysisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Demonstracja deszyfracji Podstawieniowego")
        self.root.geometry("600x750")

        self.update_queue = queue.Queue()
        self.is_running = False
        self.stop_event = None

        self.create_widgets()
        self.poll_queue()

    def update_word_count_label(self, val):
        self.word_count_label.config(text=str(int(float(val))))

    def create_widgets(self):
        frame = ttk.Frame(self.root, padding="20")
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text="Wybierz algorytm:", font=("Arial", 12, "bold")).grid(row=0, column=0, sticky="w", pady=5)

        self.algo_var = tk.StringVar(value="AG")
        self.rb_ag = ttk.Radiobutton(frame, text="Algorytm Genetyczny", variable=self.algo_var, value="AG")
        self.rb_ag.grid(row=1, column=0, sticky="w")
        self.rb_memetic = ttk.Radiobutton(frame, text="Algorytm Memetyczny (Hybryda)", variable=self.algo_var, value="MEMETIC")
        self.rb_memetic.grid(row=2, column=0, sticky="w")
        self.rb_mcmc = ttk.Radiobutton(frame, text="Monte Carlo (MCMC)", variable=self.algo_var, value="MCMC")
        self.rb_mcmc.grid(row=3, column=0, sticky="w")

        # Config frame
        config_frame = ttk.LabelFrame(frame, text="Konfiguracja", padding="10")
        config_frame.grid(row=4, column=0, sticky="ew", pady=10)

        ttk.Label(config_frame, text="Liczba słów (20-200):").grid(row=0, column=0, sticky="w")
        self.word_count_var = tk.IntVar(value=20)
        self.word_count_slider = ttk.Scale(config_frame, from_=20, to=200, orient=tk.HORIZONTAL, variable=self.word_count_var, command=self.update_word_count_label)
        self.word_count_slider.grid(row=0, column=1, sticky="ew", padx=10)
        self.word_count_label = ttk.Label(config_frame, text="20")
        self.word_count_label.grid(row=0, column=2, sticky="w")

        ttk.Label(config_frame, text="Iteracje MCMC:").grid(row=1, column=0, sticky="w", pady=5)
        self.mcmc_iter_var = tk.StringVar(value=str(config.mcmc_iterations))
        self.mcmc_iter_entry = ttk.Entry(config_frame, textvariable=self.mcmc_iter_var, width=15)
        self.mcmc_iter_entry.grid(row=1, column=1, sticky="w", padx=10, pady=5)

        ttk.Label(config_frame, text="Generacje AG:").grid(row=2, column=0, sticky="w", pady=5)
        self.ag_gen_var = tk.StringVar(value=str(config.generations))
        self.ag_gen_entry = ttk.Entry(config_frame, textvariable=self.ag_gen_var, width=15)
        self.ag_gen_entry.grid(row=2, column=1, sticky="w", padx=10, pady=5)

        ttk.Label(config_frame, text="Ilość permutacji (mutacja):").grid(row=3, column=0, sticky="w", pady=5)
        self.permutations_var = tk.StringVar(value=str(config.newKey_amount))
        self.permutations_entry = ttk.Entry(config_frame, textvariable=self.permutations_var, width=15)
        self.permutations_entry.grid(row=3, column=1, sticky="w", padx=10, pady=5)

        ttk.Label(config_frame, text="Wskaźnik mutacji (0-1):").grid(row=4, column=0, sticky="w", pady=5)
        self.mut_rate_var = tk.StringVar(value=str(config.mutation_rate))
        self.mut_rate_entry = ttk.Entry(config_frame, textvariable=self.mut_rate_var, width=15)
        self.mut_rate_entry.grid(row=4, column=1, sticky="w", padx=10, pady=5)

        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=5, column=0, pady=15, sticky="w")

        self.test_btn = ttk.Button(btn_frame, text="Test", command=self.start_test)
        self.test_btn.grid(row=0, column=0, sticky="w", padx=(0, 10))

        self.stop_btn = ttk.Button(btn_frame, text="Stop", command=self.stop_test, state=tk.DISABLED)
        self.stop_btn.grid(row=0, column=1, sticky="w")

        stats_frame = ttk.LabelFrame(frame, text="Statystyki na żywo", padding="10")
        stats_frame.grid(row=6, column=0, sticky="ew", pady=10)

        ttk.Label(stats_frame, text="Klucz szyfrujący:").grid(row=0, column=0, sticky="w")
        self.lbl_enc_key = ttk.Label(stats_frame, text="-", font=("Courier", 10))
        self.lbl_enc_key.grid(row=0, column=1, sticky="w", padx=10)

        ttk.Label(stats_frame, text="Badany klucz deszyfrujący:").grid(row=1, column=0, sticky="w")
        self.lbl_dec_key = ttk.Label(stats_frame, text="-", font=("Courier", 10))
        self.lbl_dec_key.grid(row=1, column=1, sticky="w", padx=10)

        ttk.Label(stats_frame, text="Fitness:").grid(row=2, column=0, sticky="w")
        self.lbl_fitness = ttk.Label(stats_frame, text="-")
        self.lbl_fitness.grid(row=2, column=1, sticky="w", padx=10)

        ttk.Label(stats_frame, text="Zgodność (%):").grid(row=3, column=0, sticky="w")
        self.lbl_accuracy = ttk.Label(stats_frame, text="-")
        self.lbl_accuracy.grid(row=3, column=1, sticky="w", padx=10)

        ttk.Label(stats_frame, text="Numer iteracji / pokolenia:").grid(row=4, column=0, sticky="w")
        self.lbl_iteration = ttk.Label(stats_frame, text="-")
        self.lbl_iteration.grid(row=4, column=1, sticky="w", padx=10)

        ttk.Label(frame, text="Zdekodowany tekst:").grid(row=7, column=0, sticky="w", pady=(15, 5))
        self.txt_decoded = tk.Text(frame, height=8, wrap=tk.WORD, font=("Arial", 10))
        self.txt_decoded.grid(row=8, column=0, sticky="ew")

    def stop_test(self):
        if self.stop_event:
            self.stop_event.set()
        self.stop_btn.config(state=tk.DISABLED)

    def set_ui_state(self, state):
        self.rb_ag.config(state=state)
        self.rb_memetic.config(state=state)
        self.rb_mcmc.config(state=state)
        self.word_count_slider.config(state=state)
        self.mcmc_iter_entry.config(state=state)
        self.ag_gen_entry.config(state=state)
        self.permutations_entry.config(state=state)
        self.mut_rate_entry.config(state=state)

    def start_test(self):
        if self.is_running:
            return

        try:
            config.mcmc_iterations = int(self.mcmc_iter_var.get())
        except ValueError:
            pass

        try:
            config.generations = int(self.ag_gen_var.get())
        except ValueError:
            pass

        try:
            config.newKey_amount = int(self.permutations_var.get())
        except ValueError:
            pass

        try:
            config.mutation_rate = float(self.mut_rate_var.get())
        except ValueError:
            pass
        
        word_cnt = int(self.word_count_var.get())
        config.inputText = " ".join(config.full_inputText.split()[:word_cnt])

        self.is_running = True
        self.stop_event = threading.Event()
        self.test_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.set_ui_state(tk.DISABLED)

        self.lbl_enc_key.config(text="Inicjalizacja...")
        self.lbl_dec_key.config(text="-")
        self.lbl_fitness.config(text="-")
        self.lbl_accuracy.config(text="-")
        self.lbl_iteration.config(text="-")
        self.txt_decoded.delete("1.0", tk.END)

        algo = self.algo_var.get()
        if algo == "AG":
            target_func = lambda callback: run_ag(callback=callback, stop_event=self.stop_event, use_hybrid=False)
        elif algo == "MEMETIC":
            target_func = lambda callback: run_ag(callback=callback, stop_event=self.stop_event, use_hybrid=True)
        else:
            target_func = lambda callback: run_mcmc(callback=callback, stop_event=self.stop_event)

        thread = threading.Thread(target=self.run_algo_thread, args=(target_func,), daemon=True)
        thread.start()

    def run_algo_thread(self, target_func):
        # Nowy elastyczny callback akceptujący dowolne parametry
        def callback(*args, **kwargs):
            if 'plot_data' in kwargs:
                self.update_queue.put(("PLOT", kwargs['plot_data']))
            elif len(args) == 6:
                self.update_queue.put(("UPDATE", args))
            
        try:
            target_func(callback=callback)
        except Exception as e:
            self.update_queue.put(f"Error: {e}")
        finally:
            self.update_queue.put("DONE")

    def poll_queue(self):
        try:
            while True:
                msg = self.update_queue.get_nowait()
                if msg == "DONE":
                    self.is_running = False
                    self.test_btn.config(state=tk.NORMAL)
                    self.stop_btn.config(state=tk.DISABLED)
                    self.set_ui_state(tk.NORMAL)
                elif isinstance(msg, str) and msg.startswith("Error"):
                    self.lbl_enc_key.config(text=msg)
                    self.is_running = False
                    self.test_btn.config(state=tk.NORMAL)
                    self.stop_btn.config(state=tk.DISABLED)
                    self.set_ui_state(tk.NORMAL)
                elif isinstance(msg, tuple) and msg[0] == "UPDATE":
                    enc_key, dec_key, fitness, accuracy, iteration, decoded_text = msg[1]
                    self.lbl_enc_key.config(text=enc_key)
                    self.lbl_dec_key.config(text=dec_key)
                    self.lbl_fitness.config(text=f"{fitness:.4f}")
                    self.lbl_accuracy.config(text=f"{accuracy * 100:.1f}%")
                    self.lbl_iteration.config(text=str(iteration))
                    
                    self.txt_decoded.delete("1.0", tk.END)
                    self.txt_decoded.insert("1.0", decoded_text)
                elif isinstance(msg, tuple) and msg[0] == "PLOT":
                    # Wywołujemy wykres w Wątku Głównym
                    from main_ag import plotfitness
                    plotfitness(**msg[1])
        except queue.Empty:
            pass

        self.root.after(100, self.poll_queue)

if __name__ == "__main__":
    root = tk.Tk()
    app = CryptanalysisGUI(root)
    root.mainloop()