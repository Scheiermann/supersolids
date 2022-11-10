from pathlib import Path

from sqlalchemy import create_engine, Column, Integer, String, Boolean, Float, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
import dill
import numpy as np
Base = declarative_base()

from supersolids.Schroedinger import Schroedinger
from supersolids.helper import constants


def db_create(input_path, filename_schroedinger, path_anchor_database, database_name="data.db"):
    path_database = Path(path_anchor_database, database_name)
    engine = create_engine(f"sqlite:///" + str(path_database), echo=False)
    Base.metadata.create_all(engine)
    print("Load schroedinger")
    with open(Path(input_path, filename_schroedinger), "rb") as f:
        # WARNING: this is just the input Schroedinger at t=0
        System: Schroedinger = dill.load(file=f)
    db_commit(input_path, System, path_anchor_database, database_name)


def db_commit(input_path, System: Schroedinger, frame, path_anchor_database, database_name="data.db"):
    path_database = Path(path_anchor_database, database_name)
    print(str(path_database))
    engine = create_engine(f"sqlite:///" + str(path_database), echo=False)
    if not path_database.exists():
        print("No db found. Initializing!")
        Base.metadata.create_all(engine)

    experiment_db = ExperimentDB(path=str(input_path.parent))
    system_db = SystemDB(experiment=experiment_db,
        dim=System.dim,
        imag_time=System.imag_time,
        stack_shift=System.stack_shift,
        tilt=System.tilt,
        nA_max=System.nA_max,
        max_timesteps=System.max_timesteps,
        a_s_factor=System.a_s_factor,
        a_dd_factor=System.a_dd_factor,
        dt=System.dt,
        t=System.t,
        frame=frame,
        E=System.E,
        )

    path = PathDB(input_path=str(input_path), system=system_db)

    w_db = wDB(w_x=System.w_x, w_y=System.w_y, w_z=System.w_z, system=system_db)
    mu_db = Array_mu_DB(mu_1=System.mu_arr[0], mu_2=System.mu_arr[1], system=system_db)

    a_dd_bohr = System.a_dd_array / constants.a_0
    a_s_bohr = System.a_s_array / constants.a_0

    a_dd_db = Array_a_dd_DB(a_dd_11=System.a_dd_array[0, 0], a_dd_12=System.a_dd_array[0, 1],
                            a_dd_21=System.a_dd_array[1, 0], a_dd_22=System.a_dd_array[1, 1],
                            system=system_db)

    a_dd_bohr_db = Array_a_dd_bohr_DB(a_dd_11_bohr=a_dd_bohr[0, 0], a_dd_12_bohr=a_dd_bohr[0, 1],
                                      a_dd_21_bohr=a_dd_bohr[1, 0], a_dd_22_bohr=a_dd_bohr[1, 1],
                                      system=system_db)


    a_s_db = Array_a_s_DB(a_s_11=System.a_s_array[0, 0], a_s_12=System.a_s_array[0, 1],
                          a_s_21=System.a_s_array[1, 0], a_s_22=System.a_s_array[1, 1],
                          system=system_db)

    a_s_bohr_db = Array_a_s_bohr_DB(a_s_11_bohr=a_s_bohr[0, 0], a_s_12_bohr=a_s_bohr[0, 1],
                                    a_s_21_bohr=a_s_bohr[1, 0], a_s_22_bohr=a_s_bohr[1, 1],
                                    system=system_db)


    m_u = np.array(System.m_list) / constants.u_in_kg
    m_list = List_m_DB(m_1=System.m_list[0], m_2=System.m_list[1], system=system_db)
    m_u_list = List_m_u_DB(m_1_u=m_u[0], m_2_u=m_u[1], system=system_db)

    N_list = List_N_DB(N_1=System.N_list[0], N_2=System.N_list[1], system=system_db)

    res_db = ResDB(Res_x=System.Res.x, Res_y=System.Res.y, Res_z=System.Res.z, system=system_db)
    box_db = BoxDB(Box_x0=System.Box.x0, Box_x1=System.Box.x1,
                   Box_y0=System.Box.y0, Box_y1=System.Box.y1,
                   Box_z0=System.Box.z0, Box_z1=System.Box.z1,
                   system=system_db)

    Session = sessionmaker(bind=engine)
    Session.configure(bind=engine)
    session = Session()
    session.add(system_db)
    session.add(path)
    session.commit()


def db_query(path_anchor_database, input_path, database_name="data.db"):
    path_database = Path(path_anchor_database, database_name)
    engine = create_engine(f"sqlite:///" + str(path_database), echo=False)
    Session = sessionmaker(bind=engine)
    Session.configure(bind=engine)
    session = Session()
    a = session.query(SystemDB)
    b = session.query(input_path)
    pass


class ExperimentDB(Base):
    __tablename__ = "experiments"
    id = Column(Integer, primary_key=True)
    path = Column(String)
    movies = relationship("SystemDB", back_populates="experiment")

    def __repr__(self):
        return f"<ExperimentDB(path='{self.path}')>"


class SystemDB(Base):
    __tablename__ = "systems"
    id = Column(Integer, primary_key=True)
    id_experiment = Column(Integer, ForeignKey("experiments.id"))
    dim = Column(Integer)
    imag_time = Column(Boolean)
    stack_shift = Column(Float)
    tilt = Column(Float)
    nA_max = Column(Integer)
    max_timesteps = Column(Integer)
    a_s_factor = Column(Float)
    a_dd_factor = Column(Float)
    t = Column(Float)
    E = Column(Float)
    dt = Column(Float)
    frame = Column(Integer)
    experiment = relationship("ExperimentDB", back_populates="movies")
    path_anchor = relationship("PathDB", back_populates="system", lazy='dynamic')
    Res = relationship("ResDB", back_populates="system")
    Box = relationship("BoxDB", back_populates="system")
    w = relationship("wDB", back_populates="system")

    N_list = relationship("List_N_DB", back_populates="system")

    m_list = relationship("List_m_DB", back_populates="system")
    m_u_list = relationship("List_m_u_DB", back_populates="system")

    a_s_array = relationship("Array_a_s_DB", back_populates="system")
    a_s_bohr_array = relationship("Array_a_s_bohr_DB", back_populates="system")

    a_dd_array = relationship("Array_a_dd_DB", back_populates="system")
    a_dd_bohr_array = relationship("Array_a_dd_bohr_DB", back_populates="system")

    mu_arr = relationship("Array_mu_DB", back_populates="system")

    def __repr__(self):
        return (f"<SystemDB(Res_x='{self.Res_x}', Res_y='{self.Res_y}', Res_z='{self.Res_z}',"
                + f"dim={self.dim}', imag_time='{self.imag_time}')>")


class PathDB(Base):
    __tablename__ = "system_paths"
    id = Column(Integer, primary_key=True)
    id_system = Column(Integer, ForeignKey("systems.id"))
    input_path = Column(String)
    system = relationship("SystemDB", back_populates="path_anchor")
    def __repr__(self):
        return f"<SystemPath(input_path='{self.input_path}')>"


class wDB(Base):
    __tablename__ = "ws"
    id = Column(Integer, primary_key=True)
    id_system = Column(Integer, ForeignKey("systems.id"))
    w_x = Column(Float)
    w_y = Column(Float)
    w_z = Column(Float)
    system = relationship("SystemDB", back_populates="w")
    def __repr__(self):
        return f"<wDB(w_x='{self.w_x}', w_y='{self.w_y}', w_z='{self.w_z}')>"


class ResDB(Base):
    __tablename__ = "res"
    id = Column(Integer, primary_key=True)
    id_system = Column(Integer, ForeignKey("systems.id"))
    Res_x = Column(Integer)
    Res_y = Column(Integer)
    Res_z = Column(Integer)
    system = relationship("SystemDB", back_populates="Res")
    def __repr__(self):
        return f"<ResDB(Res_x='{self.Res_x}', Res_y='{self.Res_y}', Res_z='{self.Res_z}')>"


class BoxDB(Base):
    __tablename__ = "box"
    id = Column(Integer, primary_key=True)
    id_system = Column(Integer, ForeignKey("systems.id"))
    Box_x0 = Column(Float)
    Box_x1 = Column(Float)

    Box_y0 = Column(Float)
    Box_y1 = Column(Float)

    Box_z0 = Column(Float)
    Box_z1 = Column(Float)

    system = relationship("SystemDB", back_populates="Box")
    def __repr__(self):
        return (f"<BoxDB("
                + f"Box_x0='{self.Box_x0}', Box_x1='{self.Box_x1}', "
                + f"Box_y0='{self.Box_y0}', Box_y1='{self.Box_y1}', "
                + f"Box_z0='{self.Box_z0}', Box_z1='{self.Box_z1}')>")


class List_N_DB(Base):
    __tablename__ = "N"
    id = Column(Integer, primary_key=True)
    id_system = Column(Integer, ForeignKey("systems.id"))
    N_1 = Column(Integer)
    N_2 = Column(Integer)
    system = relationship("SystemDB", back_populates="N_list")
    def __repr__(self):
        return (f"<List_N_DB([N_1='{self.N_1}', N_2='{self.N_2}')>")


class List_m_DB(Base):
    __tablename__ = "m"
    id = Column(Integer, primary_key=True)
    id_system = Column(Integer, ForeignKey("systems.id"))
    m_1 = Column(Integer)
    m_2 = Column(Integer)
    system = relationship("SystemDB", back_populates="m_list")
    def __repr__(self):
        return (f"<List_m_DB([m_1='{self.m_1}', m_2='{self.m_2}')>")


class List_m_u_DB(Base):
    __tablename__ = "m_u"
    id = Column(Integer, primary_key=True)
    id_system = Column(Integer, ForeignKey("systems.id"))
    m_1_u = Column(Integer)
    m_2_u = Column(Integer)
    system = relationship("SystemDB", back_populates="m_u_list")
    def __repr__(self):
        return (f"<List_m_DB([m_1_u='{self.m_1_u}', m_2_u='{self.m_2_u}')>")


class Array_a_s_DB(Base):
    __tablename__ = "a_s"
    id = Column(Integer, primary_key=True)
    id_system = Column(Integer, ForeignKey("systems.id"))
    a_s_11 = Column(Float)
    a_s_12 = Column(Float)
    a_s_21 = Column(Float)
    a_s_22 = Column(Float)
    system = relationship("SystemDB", back_populates="a_s_array")
    def __repr__(self):
        return (f"<Array_a_s_DB(a_s_11='{self.a_s_11}', a_s_12='{self.a_s_12}', "
                + f"a_s_21='{self.a_s_21}', a_s_22='{self.a_s_22}')>")


class Array_a_s_bohr_DB(Base):
    __tablename__ = "a_s_bohr"
    id = Column(Integer, primary_key=True)
    id_system = Column(Integer, ForeignKey("systems.id"))
    a_s_11_bohr = Column(Float)
    a_s_12_bohr = Column(Float)
    a_s_21_bohr = Column(Float)
    a_s_22_bohr = Column(Float)
    system = relationship("SystemDB", back_populates="a_s_bohr_array")
    def __repr__(self):
        return (f"<Array_a_s_bohr_DB(a_s_11_bohr='{self.a_s_11_bohr}', a_s_12_bohr='{self.a_s_12_bohr}', "
                + f"a_s_21_bohr='{self.a_s_21_bohr}', a_s_22_bohr='{self.a_s_22_bohr}')>")


class Array_a_dd_DB(Base):
    __tablename__ = "a_dd"
    id = Column(Integer, primary_key=True)
    id_system = Column(Integer, ForeignKey("systems.id"))
    a_dd_11 = Column(Float)
    a_dd_12 = Column(Float)
    a_dd_21 = Column(Float)
    a_dd_22 = Column(Float)
    system = relationship("SystemDB", back_populates="a_dd_array")
    def __repr__(self):
        return (f"<Array_a_dd_DB(a_dd_11='{self.a_dd_11}', a_dd_12='{self.a_dd_12}' "
                + f"a_dd_21='{self.a_dd_21}', a_dd_22='{self.a_dd_22}')>")


class Array_a_dd_bohr_DB(Base):
    __tablename__ = "a_dd_bohr"
    id = Column(Integer, primary_key=True)
    id_system = Column(Integer, ForeignKey("systems.id"))
    a_dd_11_bohr = Column(Float)
    a_dd_12_bohr = Column(Float)
    a_dd_21_bohr = Column(Float)
    a_dd_22_bohr = Column(Float)
    system = relationship("SystemDB", back_populates="a_dd_bohr_array")
    def __repr__(self):
        return (f"<Array_a_dd_bohr_DB(a_dd_11_bohr='{self.a_dd_11_bohr}', a_dd_12_bohr='{self.a_dd_12_bohr}' "
                + f"a_dd_21_bohr='{self.a_dd_21_bohr}', a_dd_22_bohr='{self.a_dd_22_bohr}')>")


class Array_mu_DB(Base):
    __tablename__ = "mu"
    id = Column(Integer, primary_key=True)
    id_system = Column(Integer, ForeignKey("systems.id"))
    mu_1 = Column(Float)
    mu_2 = Column(Float)
    system = relationship("SystemDB", back_populates="mu_arr")
    def __repr__(self):
        return f"<Array_mu_DB(mu_1='{self.mu_1}', mu_2='{self.mu_2}'"


# class Array_a_s_DB(Base):
#     __tablename__ = "a_s_array"
#     id = Column(Integer, primary_key=True)
#     id_system = Column(Integer, ForeignKey("systems.id"))
#     a_s_11 = Column(Float)
#     a_s_22 = Column(Float)
#     a_s_12 = Column(Float)
#     a_s_21 = Column(Float)
#     a_s_11_bohr = Column(Float)
#     a_s_22_bohr = Column(Float)
#     a_s_12_bohr = Column(Float)
#     a_s_21_bohr = Column(Float)
#     system = relationship("SystemDB", back_populates="a_s_array")
#     def __repr__(self):
#         return (f"<Array_a_s_DB([a_s_11='{self.a_s_11}', a_s_12='{self.a_s_12}', "
#                 + f"a_s_21='{self.a_s_21}', a_s_22='{self.a_s_22}'], "
#                 + f"[a_s_11_bohr='{self.a_s_11_bohr}', a_s_12_bohr='{self.a_s_12_bohr}', "
#                 + f"a_s_21_bohr='{self.a_s_21_bohr}', a_s_22_bohr='{self.a_s_22_bohr}')>")


# class Array_a_dd_DB(Base):
#     __tablename__ = "a_dd_array"
#     id = Column(Integer, primary_key=True)
#     id_system = Column(Integer, ForeignKey("systems.id"))
#     a_dd_11 = Column(Float)
#     a_dd_22 = Column(Float)
#     a_dd_12 = Column(Float)
#     a_dd_21 = Column(Float)
#     a_dd_11_bohr = Column(Float)
#     a_dd_22_bohr = Column(Float)
#     a_dd_12_bohr = Column(Float)
#     a_dd_21_bohr = Column(Float)
#     system = relationship("SystemDB", back_populates="a_dd_array")
#     def __repr__(self):
#         return (f"<Array_a_dd_DB("
#                 + f"[a_dd_11='{self.a_dd_11}', a_dd_12='{self.a_dd_12}', "
#                 + f"a_dd_21='{self.a_dd_21}', a_dd_22='{self.a_dd_22}'], "
#                 + f"[a_dd_11_bohr='{self.a_dd_11_bohr}', a_dd_12_bohr='{self.a_dd_12_bohr}', "
#                 + f"a_dd_21_bohr='{self.a_dd_21_bohr}', a_dd_22_bohr='{self.a_dd_22_bohr}'])>")