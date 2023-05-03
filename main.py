import streamlit as st
from app1 import main as app1 
from app2 import main as app2


def main():
    pag_name = ["EDA", "INFERENCE"]

    sim_selection = st.selectbox('Seleziona la pagina', pag_name)

    if sim_selection == pag_name[0]:
        app1()
    elif sim_selection == pag_name[1]:
        app2()
    else:
        st.markdown("Something went wrong. We are looking into it.")

    
    
    
    
    
if __name__ == '__main__':
    main()
