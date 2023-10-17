"""empty message

Revision ID: 1b4cfe9d390a
Revises: 
Create Date: 2023-10-17 12:29:50.122507

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel
import app


# revision identifiers, used by Alembic.
revision: str = '1b4cfe9d390a'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('cachedresponse',
    sa.Column('created', app.db.base.TZDateTime(timezone=True), nullable=True),
    sa.Column('updated', app.db.base.TZDateTime(timezone=True), nullable=True),
    sa.Column('messages', sa.JSON(), nullable=True),
    sa.Column('response', sa.JSON(), nullable=True),
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('hash', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('lens_type', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('functions', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
    sa.Column('model', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('version', sa.Integer(), nullable=False),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('hash', 'lens_type', 'model', 'version', name='unique_hash_model_version')
    )
    op.create_index(op.f('ix_cachedresponse_hash'), 'cachedresponse', ['hash'], unique=False)
    op.create_index(op.f('ix_cachedresponse_id'), 'cachedresponse', ['id'], unique=False)
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index(op.f('ix_cachedresponse_id'), table_name='cachedresponse')
    op.drop_index(op.f('ix_cachedresponse_hash'), table_name='cachedresponse')
    op.drop_table('cachedresponse')
    # ### end Alembic commands ###