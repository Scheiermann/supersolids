from pathlib import Path

from sqlalchemy import create_engine, Column, Integer, String, Boolean, Float, ForeignKey, select
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

def db_check():
    test = "/bigwork/dscheier/results/begin_ramp_11_08_eq_65_70_75_80_85_90"
    path_database = "/bigwork/dscheier/results/begin_ramp_11_08_eq_65_70_75_80_85_90/data.db"
    engine = create_engine(f"sqlite:///" + str(path_database), echo=False)
    Session = sessionmaker(bind=engine)
    Session.configure(bind=engine)
    session = Session()
    # experiment_db = session.query(ExperimentDB.id).filter_by(path=test)
    experiment_db = get_or_create(session, ExperimentDB, path=test)
    print(experiment_db)


def db_commit(experiment_name, input_path, System: Schroedinger, frame, path_anchor_database, database_name="data.db"):
    path_database = Path(path_anchor_database, database_name)
    print(f"Commit entry to {str(path_database)}")
    engine = create_engine(f"sqlite:///" + str(path_database), echo=False)
    if not path_database.exists():
        print("No db found. Initializing!")
        Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    Session.configure(bind=engine)
    session = Session()

    experiment_db, _ = get_or_create(session, ExperimentDB, name=experiment_name)
    path_experiment_db, _ = get_or_create(session, PathExperimentDB,
                                          path=str(input_path.parent),
                                          id_experiment=experiment_db.id,
                                          relation={"experiment":experiment_db},
                                          )
    system_db, _ = get_or_create(session, SystemDB,
        dim=System.dim,
        imag_time=System.imag_time,
        stack_shift=System.stack_shift,
        tilt=System.tilt,
        nA_max=System.nA_max,
        max_timesteps=System.max_timesteps,
        a_s_factor=System.a_s_factor,
        a_dd_factor=System.a_dd_factor,
        dt=System.dt,
        path=str(input_path),
        relation={"path_exp":path_experiment_db},
        )

    # a = Association()
    # a.system_tab = system_db
    # path_experiment_db.system.append(a)
                                             
    # frames_db = FramesDB(frame=frame, t=System.t, E=System.E, system=system_db)
    frames_db, _ = get_or_create(session, FramesDB,
                                 frame=frame, t=System.t, E=System.E,
                                 system=system_db,
                                 relation={"system":system_db},
                                 )

    # w_db = wDB(w_x=System.w_x, w_y=System.w_y, w_z=System.w_z, system=system_db)
    w_db, _ = get_or_create(session, wDB,
                            w_x=System.w_x, w_y=System.w_y, w_z=System.w_z,
                            system=system_db,
                            relation={"system":system_db},
                            )


    # mu_db = Array_mu_DB(mu_1=System.mu_arr[0], mu_2=System.mu_arr[1], system=system_db)
    mu_db, _ = get_or_create(session, Array_mu_DB,
                            mu_1=System.mu_arr[0], mu_2=System.mu_arr[1],
                            system=system_db,
                            relation={"system":system_db},
                            )


    l_0 = np.sqrt(constants.hbar / (System.m_list[0] * System.w_x))
    a_dd_bohr = (System.a_dd_array / constants.a_0) * l_0
    a_s_bohr = (System.a_s_array / constants.a_0) * l_0


    # a_dd_db = Array_a_dd_DB(a_dd_11=System.a_dd_array[0, 0], a_dd_12=System.a_dd_array[0, 1],
    #                         a_dd_21=System.a_dd_array[1, 0], a_dd_22=System.a_dd_array[1, 1],
    #                         system=system_db)
    a_dd_db, _ = get_or_create(session, Array_a_dd_DB,
                               a_dd_11=System.a_dd_array[0, 0], a_dd_12=System.a_dd_array[0, 1],
                               a_dd_21=System.a_dd_array[1, 0], a_dd_22=System.a_dd_array[1, 1],
                               system=system_db,
                               relation={"system":system_db},
                               )


    # a_dd_bohr_db = Array_a_dd_bohr_DB(a_dd_11_bohr=a_dd_bohr[0, 0], a_dd_12_bohr=a_dd_bohr[0, 1],
    #                                   a_dd_21_bohr=a_dd_bohr[1, 0], a_dd_22_bohr=a_dd_bohr[1, 1],
    #                                   system=system_db)
    a_dd_bohr_db, _ = get_or_create(session, Array_a_dd_bohr_DB,
                                    a_dd_11_bohr=a_dd_bohr[0, 0], a_dd_12_bohr=a_dd_bohr[0, 1],
                                    a_dd_21_bohr=a_dd_bohr[1, 0], a_dd_22_bohr=a_dd_bohr[1, 1],
                                    system=system_db,
                                    relation={"system":system_db},
                                    )


    # a_s_db = Array_a_s_DB(a_s_11=System.a_s_array[0, 0], a_s_12=System.a_s_array[0, 1],
    #                       a_s_21=System.a_s_array[1, 0], a_s_22=System.a_s_array[1, 1],
    #                       system=system_db)
    a_s_db, _ = get_or_create(session, Array_a_s_DB,
                              a_s_11=System.a_s_array[0, 0], a_s_12=System.a_s_array[0, 1],
                              a_s_21=System.a_s_array[1, 0], a_s_22=System.a_s_array[1, 1],
                              system=system_db,
                              relation={"system":system_db},
                              )


    # a_s_bohr_db = Array_a_s_bohr_DB(a_s_11_bohr=a_s_bohr[0, 0], a_s_12_bohr=a_s_bohr[0, 1],
    #                                 a_s_21_bohr=a_s_bohr[1, 0], a_s_22_bohr=a_s_bohr[1, 1],
    #                                 system=system_db)
    a_s_bohr_db, _ = get_or_create(session, Array_a_s_bohr_DB,
                                   a_s_11_bohr=a_s_bohr[0, 0], a_s_12_bohr=a_s_bohr[0, 1],
                                   a_s_21_bohr=a_s_bohr[1, 0], a_s_22_bohr=a_s_bohr[1, 1],
                                   system=system_db,
                                   relation={"system":system_db},
                                   )

    m_u = np.array(System.m_list) / constants.u_in_kg
    # m_list = List_m_DB(m_1=System.m_list[0], m_2=System.m_list[1], system=system_db)
    m_list, _ = get_or_create(session, List_m_DB,
                              m_1=System.m_list[0], m_2=System.m_list[1],
                              system=system_db,
                              relation={"system":system_db},
                              )


    # m_u_list = List_m_u_DB(m_1_u=m_u[0], m_2_u=m_u[1], system=system_db)
    m_u_list, _ = get_or_create(session, List_m_u_DB,
                                m_1_u=m_u[0], m_2_u=m_u[1],
                                system=system_db,
                                relation={"system":system_db},
                                )


    # N_list = List_N_DB(N_1=System.N_list[0], N_2=System.N_list[1], system=system_db)
    N_list, _ = get_or_create(session, List_N_DB,
                              N_1=System.N_list[0], N_2=System.N_list[1],
                              system=system_db,
                              relation={"system":system_db},
                              )


    # res_db = ResDB(Res_x=System.Res.x, Res_y=System.Res.y, Res_z=System.Res.z, system=system_db)
    res_db, _ = get_or_create(session, ResDB,
                              Res_x=System.Res.x, Res_y=System.Res.y, Res_z=System.Res.z,
                              system=system_db,
                              relation={"system":system_db},
                              )


    # box_db = BoxDB(Box_x0=System.Box.x0, Box_x1=System.Box.x1,
    #                Box_y0=System.Box.y0, Box_y1=System.Box.y1,
    #                Box_z0=System.Box.z0, Box_z1=System.Box.z1,
    #                system=system_db)
    box_db, _ = get_or_create(session, BoxDB,
                              Box_x0=System.Box.x0, Box_x1=System.Box.x1,
                              Box_y0=System.Box.y0, Box_y1=System.Box.y1,
                              Box_z0=System.Box.z0, Box_z1=System.Box.z1,
                              system=system_db,
                              relation={"system":system_db},
                              )


    session.add_all([experiment_db, path_experiment_db, system_db, frames_db,
                     w_db, mu_db, a_dd_db, a_dd_bohr_db, a_s_db, a_s_bohr_db,
                     m_list, m_u_list, N_list, res_db, box_db
                    ])
    session.commit()


def db_query(path_anchor_database, database_name="data.db"):
    path_database = Path(path_anchor_database, database_name)
    engine = create_engine(f"sqlite:///" + str(path_database), echo=False)
    Session = sessionmaker(bind=engine)
    Session.configure(bind=engine)
    session = Session()
    a = session.query(SystemDB)
    session.query(SystemDB, wDB, ResDB, BoxDB, List_N_DB, List_m_u_DB,
                  Array_mu_DB, Array_a_dd_bohr_DB, Array_a_s_bohr_DB). \
            filter(SystemDB.id == wDB.id_system). \
            filter(SystemDB.id == ResDB.id_system). \
            filter(SystemDB.id == BoxDB.id_system). \
            filter(SystemDB.id == BoxDB.id_system). \
            filter(SystemDB.id == wDB.id_system). \
            filter(SystemDB.id == List_N_DB.id_system). \
            filter(SystemDB.id == List_m_u_DB.id_system). \
            filter(SystemDB.id == Array_mu_DB.id_system). \
            filter(SystemDB.id == Array_a_dd_bohr_DB.id_system). \
            filter(SystemDB.id == Array_a_s_bohr_DB.id_system). \
            all()
    pass


def get_or_create(session, model, defaults=None, relation=None, **kwargs):
    if relation:
        new_kwargs = {**kwargs, **relation}
    else:
        new_kwargs=kwargs

    instance = session.query(model).filter_by(**kwargs).one_or_none()

    if instance:
        return instance, False
    else:
        kwargs |= defaults or {}
        instance = model(**new_kwargs)
        try:
            session.add(instance)
            session.commit()
        except Exception:
            session.rollback()
            instance = session.query(model).filter_by(**kwargs).one()
            return instance, False
        else:
            return instance, True


class ExperimentDB(Base):
    __tablename__ = "experiments"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    path_experiment = relationship("PathExperimentDB", back_populates="experiment")

    def __repr__(self):
        return f"<ExperimentDB(name='{self.name}')>"


# class Association(Base):
#     __tablename__ = "association_table"
#     path_experiments_id = Column(ForeignKey("path_experiments.id"), primary_key=True)
#     systems_id = Column(ForeignKey("systems.id"), primary_key=True)
#     system_tab = relationship("SystemDB", back_populates="path_exp")
#     path_exp_tab = relationship("PathExperimentDB", back_populates="system")



class PathExperimentDB(Base):
    __tablename__ = "path_experiments"
    id = Column(Integer, primary_key=True)
    id_experiment = Column(Integer, ForeignKey("experiments.id"))
    path = Column(String)
    experiment = relationship("ExperimentDB", back_populates="path_experiment")
    # system = relationship("Association", back_populates="path_exp_tab")
    system = relationship("SystemDB", back_populates="path_exp")
    def __repr__(self):
        return f"<PathExperiment(path='{self.path}')>"


class SystemDB(Base):
    __tablename__ = "systems"
    id = Column(Integer, primary_key=True)
    id_path_experiment = Column(Integer, ForeignKey("path_experiments.id"))
    dim = Column(Integer)
    imag_time = Column(Boolean)
    stack_shift = Column(Float)
    tilt = Column(Float)
    nA_max = Column(Integer)
    max_timesteps = Column(Integer)
    a_s_factor = Column(Float)
    a_dd_factor = Column(Float)
    dt = Column(Float)
    path = Column(String)

    # path_exp = relationship("Association", back_populates="system_tab")
    path_exp = relationship("PathExperimentDB", back_populates="system")

    frames = relationship("FramesDB", back_populates="system")

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
        return (f"<SystemDB(path='{self.path}', dim='{self.dim}', imag_time='{self.imag_time}')>")


class FramesDB(Base):
    __tablename__ = "frames"
    id = Column(Integer, primary_key=True)
    id_system = Column(Integer, ForeignKey("systems.id"))
    frame = Column(Integer)
    t = Column(Float)
    E = Column(Float)

    system = relationship("SystemDB", back_populates="frames")
    def __repr__(self):
        return (f"<Frames(" + f"frame='{self.frame}', t='{self.t}', " + f"E='{self.E}')>")


class wDB(Base):
    __tablename__ = "w"
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